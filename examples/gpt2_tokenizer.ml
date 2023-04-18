(* merge file: https://cdn.huggingface.co/distilgpt2-merges.txt *)
(* TODO: Unicode is not properly supported here. *)
open! Base

module String_pair = struct
  module T = struct
    type t = string * string [@@deriving sexp]

    let compare (s1, s1') (s2, s2') =
      let cmp = String.compare s1 s2 in
      if cmp <> 0 then cmp else String.compare s1' s2'
  end

  include T
  include Comparator.Make (T)
end

let unicode_chars str =
  let decoder = Uutf.decoder (`String str) in
  let rec loop acc ~prev_start =
    match Uutf.decode decoder with
    | `Uchar _ ->
      let next_start = Uutf.decoder_byte_count decoder in
      let char = String.sub str ~pos:prev_start ~len:(next_start - prev_start) in
      loop (char :: acc) ~prev_start:next_start
    | `End | `Malformed _ -> List.rev acc
    | `Await -> assert false
  in
  loop [] ~prev_start:0

let bytes_to_unicode =
  [ 33, "!"
  ; 34, "\""
  ; 35, "#"
  ; 36, "$"
  ; 37, "%"
  ; 38, "&"
  ; 39, "\'"
  ; 40, "("
  ; 41, ")"
  ; 42, "*"
  ; 43, "+"
  ; 44, ","
  ; 45, "-"
  ; 46, "."
  ; 47, "/"
  ; 48, "0"
  ; 49, "1"
  ; 50, "2"
  ; 51, "3"
  ; 52, "4"
  ; 53, "5"
  ; 54, "6"
  ; 55, "7"
  ; 56, "8"
  ; 57, "9"
  ; 58, ":"
  ; 59, ";"
  ; 60, "<"
  ; 61, "="
  ; 62, ">"
  ; 63, "?"
  ; 64, "@"
  ; 65, "A"
  ; 66, "B"
  ; 67, "C"
  ; 68, "D"
  ; 69, "E"
  ; 70, "F"
  ; 71, "G"
  ; 72, "H"
  ; 73, "I"
  ; 74, "J"
  ; 75, "K"
  ; 76, "L"
  ; 77, "M"
  ; 78, "N"
  ; 79, "O"
  ; 80, "P"
  ; 81, "Q"
  ; 82, "R"
  ; 83, "S"
  ; 84, "T"
  ; 85, "U"
  ; 86, "V"
  ; 87, "W"
  ; 88, "X"
  ; 89, "Y"
  ; 90, "Z"
  ; 91, "["
  ; 92, "\\"
  ; 93, "]"
  ; 94, "^"
  ; 95, "_"
  ; 96, "`"
  ; 97, "a"
  ; 98, "b"
  ; 99, "c"
  ; 100, "d"
  ; 101, "e"
  ; 102, "f"
  ; 103, "g"
  ; 104, "h"
  ; 105, "i"
  ; 106, "j"
  ; 107, "k"
  ; 108, "l"
  ; 109, "m"
  ; 110, "n"
  ; 111, "o"
  ; 112, "p"
  ; 113, "q"
  ; 114, "r"
  ; 115, "s"
  ; 116, "t"
  ; 117, "u"
  ; 118, "v"
  ; 119, "w"
  ; 120, "x"
  ; 121, "y"
  ; 122, "z"
  ; 123, "{"
  ; 124, "|"
  ; 125, "}"
  ; 126, "~"
  ; 161, "¡"
  ; 162, "¢"
  ; 163, "£"
  ; 164, "¤"
  ; 165, "¥"
  ; 166, "¦"
  ; 167, "§"
  ; 168, "¨"
  ; 169, "©"
  ; 170, "ª"
  ; 171, "«"
  ; 172, "¬"
  ; 174, "®"
  ; 175, "¯"
  ; 176, "°"
  ; 177, "±"
  ; 178, "²"
  ; 179, "³"
  ; 180, "´"
  ; 181, "µ"
  ; 182, "¶"
  ; 183, "·"
  ; 184, "¸"
  ; 185, "¹"
  ; 186, "º"
  ; 187, "»"
  ; 188, "¼"
  ; 189, "½"
  ; 190, "¾"
  ; 191, "¿"
  ; 192, "À"
  ; 193, "Á"
  ; 194, "Â"
  ; 195, "Ã"
  ; 196, "Ä"
  ; 197, "Å"
  ; 198, "Æ"
  ; 199, "Ç"
  ; 200, "È"
  ; 201, "É"
  ; 202, "Ê"
  ; 203, "Ë"
  ; 204, "Ì"
  ; 205, "Í"
  ; 206, "Î"
  ; 207, "Ï"
  ; 208, "Ð"
  ; 209, "Ñ"
  ; 210, "Ò"
  ; 211, "Ó"
  ; 212, "Ô"
  ; 213, "Õ"
  ; 214, "Ö"
  ; 215, "×"
  ; 216, "Ø"
  ; 217, "Ù"
  ; 218, "Ú"
  ; 219, "Û"
  ; 220, "Ü"
  ; 221, "Ý"
  ; 222, "Þ"
  ; 223, "ß"
  ; 224, "à"
  ; 225, "á"
  ; 226, "â"
  ; 227, "ã"
  ; 228, "ä"
  ; 229, "å"
  ; 230, "æ"
  ; 231, "ç"
  ; 232, "è"
  ; 233, "é"
  ; 234, "ê"
  ; 235, "ë"
  ; 236, "ì"
  ; 237, "í"
  ; 238, "î"
  ; 239, "ï"
  ; 240, "ð"
  ; 241, "ñ"
  ; 242, "ò"
  ; 243, "ó"
  ; 244, "ô"
  ; 245, "õ"
  ; 246, "ö"
  ; 247, "÷"
  ; 248, "ø"
  ; 249, "ù"
  ; 250, "ú"
  ; 251, "û"
  ; 252, "ü"
  ; 253, "ý"
  ; 254, "þ"
  ; 255, "ÿ"
  ; 0, "Ā"
  ; 1, "ā"
  ; 2, "Ă"
  ; 3, "ă"
  ; 4, "Ą"
  ; 5, "ą"
  ; 6, "Ć"
  ; 7, "ć"
  ; 8, "Ĉ"
  ; 9, "ĉ"
  ; 10, "Ċ"
  ; 11, "ċ"
  ; 12, "Č"
  ; 13, "č"
  ; 14, "Ď"
  ; 15, "ď"
  ; 16, "Đ"
  ; 17, "đ"
  ; 18, "Ē"
  ; 19, "ē"
  ; 20, "Ĕ"
  ; 21, "ĕ"
  ; 22, "Ė"
  ; 23, "ė"
  ; 24, "Ę"
  ; 25, "ę"
  ; 26, "Ě"
  ; 27, "ě"
  ; 28, "Ĝ"
  ; 29, "ĝ"
  ; 30, "Ğ"
  ; 31, "ğ"
  ; 32, "Ġ"
  ; 127, "ġ"
  ; 128, "Ģ"
  ; 129, "ģ"
  ; 130, "Ĥ"
  ; 131, "ĥ"
  ; 132, "Ħ"
  ; 133, "ħ"
  ; 134, "Ĩ"
  ; 135, "ĩ"
  ; 136, "Ī"
  ; 137, "ī"
  ; 138, "Ĭ"
  ; 139, "ĭ"
  ; 140, "Į"
  ; 141, "į"
  ; 142, "İ"
  ; 143, "ı"
  ; 144, "Ĳ"
  ; 145, "ĳ"
  ; 146, "Ĵ"
  ; 147, "ĵ"
  ; 148, "Ķ"
  ; 149, "ķ"
  ; 150, "ĸ"
  ; 151, "Ĺ"
  ; 152, "ĺ"
  ; 153, "Ļ"
  ; 154, "ļ"
  ; 155, "Ľ"
  ; 156, "ľ"
  ; 157, "Ŀ"
  ; 158, "ŀ"
  ; 159, "Ł"
  ; 160, "ł"
  ; 173, "Ń"
  ]

(*
let pattern_re =
  {|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+|}
  *)
let pattern_re = {|'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\d+| ?[^\s\[a-zA-Z]\d]+|\s+|}

type t =
  { re : Re.re
  ; encoder : (string, int, String.comparator_witness) Map.t
  ; decoder : string array
  ; bpe_ranks : (String_pair.t, int, String_pair.comparator_witness) Map.t
  ; start_of_text_token : int
  ; end_of_text_token : int
  }

let create ~merge_filename =
  let u_to_byte =
    List.map bytes_to_unicode ~f:(fun (b, u) -> u, b) |> Map.of_alist_exn (module String)
  in
  let conv_bpe str =
    unicode_chars str
    |> List.filter_map ~f:(fun s -> Map.find u_to_byte s |> Option.map ~f:Char.of_int_exn)
    |> String.of_char_list
  in
  let bpe_lines =
    Stdio.In_channel.read_lines merge_filename
    |> List.tl_exn
    |> List.mapi ~f:(fun idx line ->
         String.strip line
         |> String.split ~on:' '
         |> function
         | [ str1; str2 ] -> conv_bpe str1, conv_bpe str2
         | _ -> Printf.failwithf "multiple space characters in line %d: %s" idx line ())
  in
  let vocab = Queue.create () in
  List.iter bytes_to_unicode ~f:(fun (b, _u) ->
    Char.of_int_exn b |> String.of_char |> Queue.enqueue vocab);
  List.iter bpe_lines ~f:(fun (s1, s2) -> Queue.enqueue vocab (s1 ^ s2));
  let end_of_text_token = Queue.length vocab in
  Queue.enqueue vocab "<|endoftext|>";
  let decoder = Queue.to_array vocab in
  let encoder =
    Array.to_list decoder
    |> List.mapi ~f:(fun i v -> v, i)
    |> Map.of_alist_exn (module String)
  in
  let bpe_ranks =
    List.mapi bpe_lines ~f:(fun i p -> p, i) |> Map.of_alist_exn (module String_pair)
  in
  { re = Re.Perl.re pattern_re |> Re.compile
  ; encoder
  ; decoder
  ; bpe_ranks
  ; start_of_text_token = end_of_text_token
  ; end_of_text_token
  }

let encode _t _str = failwith "TODO"
let decode t ids = List.map ids ~f:(fun i -> t.decoder.(i)) |> String.concat ~sep:""
