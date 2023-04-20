open! Import
open! Base

exception Cannot_write
exception Read_error of string

let read_error fmt = Printf.ksprintf (fun s -> raise (Read_error s)) fmt
let magic_string = "\147NUMPY"
let magic_string_len = String.length magic_string

let dtype ~ty =
  let endianness = if Sys.big_endian then ">" else "<" in
  let kind =
    match (ty : Element_type.t) with
    | S8 -> "i1"
    | S16 -> "i2"
    | S32 -> "i4"
    | S64 -> "i8"
    | U8 -> "u1"
    | U16 -> "u2"
    | U32 -> "u4"
    | U64 -> "u8"
    | F32 -> "f4"
    | F64 -> "f8"
    | C64 -> "c8" (* 2 32bits float. *)
    | C128 -> "c16" (* 2 64bits float. *)
    | ty -> [%message "unsupported" (ty : Element_type.t)] |> failwith_s
  in
  endianness ^ kind

let shape ~dims =
  match dims with
  | [ dim1 ] -> Printf.sprintf "%d," dim1
  | dims -> List.map dims ~f:Int.to_string |> String.concat ~sep:", "

let full_header ?header_len ~ty ~dims () =
  let header =
    Printf.sprintf
      "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }"
      (dtype ~ty)
      (shape ~dims)
  in
  let padding_len =
    let total_len = String.length header + magic_string_len + 4 + 1 in
    match header_len with
    | None -> if total_len % 16 = 0 then 0 else 16 - (total_len % 16)
    | Some header_len ->
      if header_len % 16 <> 0
      then
        [%message "header_len has to be divisible by 16" (header_len : int)] |> failwith_s;
      if header_len < total_len
      then
        [%message
          "header_len is smaller than total_len" (header_len : int) (total_len : int)]
        |> failwith_s;
      header_len - total_len
  in
  let total_header_len = String.length header + padding_len + 1 in
  Printf.sprintf
    "%s\001\000%c%c%s%s\n"
    magic_string
    (total_header_len % 256 |> Char.of_int_exn)
    (total_header_len / 256 |> Char.of_int_exn)
    header
    (String.make padding_len ' ')

let with_file filename flags mask ~f =
  let file_descr = Unix.openfile filename flags mask in
  try
    let result = f file_descr in
    Unix.close file_descr;
    result
  with
  | exn ->
    Unix.close file_descr;
    raise exn

let map_file file_descr ~pos kind layout shared shape =
  let is_scalar = Array.length shape = 0 in
  let array =
    Unix.map_file
      file_descr
      ~pos
      kind
      layout
      shared
      (if is_scalar then [| 1 |] else shape)
  in
  if is_scalar then Bigarray.reshape array [||] else array

let write ?header_len literal filename =
  let shape = Literal.shape literal in
  let ty = Shape.ty shape in
  let dims = Shape.dimensions shape in
  let (Element_type.P kind) =
    match Element_type.ba_kind ty with
    | Some kind -> kind
    | None -> [%message "unsupported" (ty : Element_type.t)] |> failwith_s
  in
  with_file filename [ O_CREAT; O_TRUNC; O_RDWR ] 0o640 ~f:(fun file_descr ->
    let full_header = full_header () ?header_len ~ty ~dims in
    let full_header_len = String.length full_header in
    if Unix.write_substring file_descr full_header 0 full_header_len <> full_header_len
    then raise Cannot_write;
    let file_array =
      map_file
        ~pos:(Int64.of_int full_header_len)
        file_descr
        kind
        C_layout
        true
        (Array.of_list dims)
    in
    Literal.copy_to_bigarray literal ~dst:file_array)

let really_read fd len =
  let buffer = Bytes.create len in
  let rec loop offset =
    let read = Unix.read fd buffer offset (len - offset) in
    if read + offset < len
    then loop (read + offset)
    else if read = 0
    then read_error "unexpected eof"
  in
  loop 0;
  Bytes.to_string buffer

module Header = struct
  type packed_kind = P : (_, _) Bigarray.kind -> packed_kind

  type t =
    { kind : packed_kind
    ; fortran_order : bool
    ; shape : int array
    }

  let split str ~on =
    let parens = ref 0 in
    let indexes = ref [] in
    for i = 0 to String.length str - 1 do
      match str.[i] with
      | '(' -> Int.incr parens
      | ')' -> Int.decr parens
      | c when !parens = 0 && Char.( = ) c on -> indexes := i :: !indexes
      | _ -> ()
    done;
    List.fold
      ~init:(String.length str, [])
      ~f:(fun (prev_p, acc) index ->
        index, String.sub str ~pos:(index + 1) ~len:(prev_p - index - 1) :: acc)
      !indexes
    |> fun (first_pos, acc) -> String.sub str ~pos:0 ~len:first_pos :: acc

  let trim str ~on =
    let rec loopr start len =
      if len = 0
      then start, len
      else if List.mem on str.[start + len - 1] ~equal:Char.equal
      then loopr start (len - 1)
      else start, len
    in
    let rec loopl start len =
      if len = 0
      then start, len
      else if List.mem on str.[start] ~equal:Char.equal
      then loopl (start + 1) (len - 1)
      else loopr start len
    in
    let start, len = loopl 0 (String.length str) in
    String.sub str ~pos:start ~len

  let parse header =
    let header_fields =
      trim header ~on:[ '{'; ' '; '}'; '\n' ]
      |> split ~on:','
      |> List.map ~f:Caml.String.trim
      |> List.filter ~f:(fun s -> String.length s > 0)
      |> List.map ~f:(fun header_field ->
           match split header_field ~on:':' with
           | [ name; value ] ->
             trim name ~on:[ '\''; ' ' ], trim value ~on:[ '\''; ' '; '('; ')' ]
           | _ -> read_error "unable to parse field %s" header_field)
    in
    let find_field field =
      match List.Assoc.find header_fields field ~equal:String.equal with
      | Some v -> v
      | None -> read_error "cannot find field %s" field
    in
    let kind =
      let kind = find_field "descr" in
      (match kind.[0] with
       | '|' | '=' -> ()
       | '>' ->
         if not Sys.big_endian then read_error "big endian data but arch is little endian"
       | '<' ->
         if Sys.big_endian then read_error "little endian data but arch is big endian"
       | otherwise -> read_error "incorrect endianness %c" otherwise);
      match String.sub kind ~pos:1 ~len:(String.length kind - 1) with
      | "f4" -> P Float32
      | "f8" -> P Float64
      | "i4" -> P Int32
      | "i8" -> P Int64
      | "u1" -> P Int8_unsigned
      | "i1" -> P Int8_signed
      | "u2" -> P Int16_unsigned
      | "i2" -> P Int16_signed
      | "S1" -> P Char
      | "c8" -> P Complex32
      | "c16" -> P Complex64
      | otherwise -> read_error "incorrect descr %s" otherwise
    in
    let fortran_order =
      match find_field "fortran_order" with
      | "False" -> false
      | "True" -> true
      | otherwise -> read_error "incorrect fortran_order %s" otherwise
    in
    let shape =
      find_field "shape"
      |> split ~on:','
      |> List.map ~f:Caml.String.trim
      |> List.filter ~f:(fun s -> String.length s > 0)
      |> List.map ~f:Int.of_string
      |> Array.of_list
    in
    { kind; fortran_order; shape }
end

type packed_array = P : (_, _, _) Bigarray.Genarray.t -> packed_array

let read_mmap filename ~shared =
  let access = if shared then Unix.O_RDWR else O_RDONLY in
  let file_descr = Unix.openfile filename [ access ] 0 in
  let pos, header =
    try
      let magic_string' = really_read file_descr magic_string_len in
      if String.( <> ) magic_string magic_string' then read_error "magic string mismatch";
      let version = really_read file_descr 2 |> fun v -> v.[0] |> Char.to_int in
      let header_len_len =
        match version with
        | 1 -> 2
        | 2 -> 4
        | _ -> read_error "unsupported version %d" version
      in
      let header, header_len =
        really_read file_descr header_len_len
        |> fun str ->
        let header_len = ref 0 in
        for i = String.length str - 1 downto 0 do
          header_len := (256 * !header_len) + Char.to_int str.[i]
        done;
        really_read file_descr !header_len, !header_len
      in
      let header = Header.parse header in
      Int64.of_int (header_len + header_len_len + magic_string_len + 2), header
    with
    | exn ->
      Unix.close file_descr;
      raise exn
  in
  let (Header.P kind) = header.kind in
  let build layout =
    let array = map_file file_descr ~pos kind layout shared header.shape in
    Caml.Gc.finalise (fun _ -> Unix.close file_descr) array;
    P array
  in
  if header.fortran_order then build Fortran_layout else build C_layout

let read filename =
  let (P array) = read_mmap filename ~shared:false in
  match Bigarray.Genarray.layout array with
  | C_layout -> Literal.of_bigarray array
  | Fortran_layout -> [%message "fortran layout not supported" filename] |> failwith_s

module Npz = struct
  let npy_suffix = ".npy"

  let maybe_add_suffix array_name ~suffix =
    let suffix =
      match suffix with
      | None -> npy_suffix
      | Some suffix -> suffix
    in
    array_name ^ suffix

  type in_file = Zip.in_file

  let open_in = Zip.open_in

  let entries t =
    Zip.entries t
    |> List.map ~f:(fun entry ->
         let filename = entry.Zip.filename in
         if String.length filename < String.length npy_suffix
         then filename
         else (
           let start_pos = String.length filename - String.length npy_suffix in
           if String.( = )
                (String.sub filename ~pos:start_pos ~len:(String.length npy_suffix))
                npy_suffix
           then String.sub filename ~pos:0 ~len:start_pos
           else filename))

  let close_in = Zip.close_in

  let read ?suffix t array_name =
    let array_name = maybe_add_suffix array_name ~suffix in
    let entry =
      try Zip.find_entry t array_name with
      | Caml.Not_found -> raise (Invalid_argument ("unable to find " ^ array_name))
    in
    let tmp_file = Caml.Filename.temp_file "ocaml-npz" ".tmp" in
    Zip.copy_entry_to_file t entry tmp_file;
    let data = read tmp_file in
    Caml.Sys.remove tmp_file;
    data

  let read_all filename =
    let t = open_in filename in
    Exn.protect
      ~f:(fun () ->
        entries t
        |> List.map ~f:(fun entry -> entry, read t entry)
        |> Hashtbl.of_alist_exn (module String))
      ~finally:(fun () -> close_in t)

  type out_file = Zip.out_file

  let open_out filename = Zip.open_out filename
  let close_out = Zip.close_out

  let write ?suffix t array_name array =
    let array_name = maybe_add_suffix array_name ~suffix in
    let tmp_file = Caml.Filename.temp_file "ocaml-npz" ".tmp" in
    write array tmp_file;
    Zip.copy_file_to_entry tmp_file t array_name;
    Caml.Sys.remove tmp_file
end
