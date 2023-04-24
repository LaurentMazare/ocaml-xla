(* Very naive implementation of the sentencepiece tokenizer, this only supports
   BPE models and not the unigram ones. *)
open! Base

let failwith_s s = Sexp.to_string s |> failwith

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

let delim = "▁"

type t =
  { encoder : (string, int, String.comparator_witness) Map.t
  ; decoder : (int, string, Int.comparator_witness) Map.t
  ; bpe_ranks : (String_pair.t, int, String_pair.comparator_witness) Map.t
  }

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

(* byte-level Byte-Pair-Encoding *)
let bpe t str =
  let rec loop words =
    let rec best_pair words min_rank =
      match words with
      | [] | [ _ ] -> min_rank
      | head1 :: (head2 :: _ as words) ->
        let pair = head1, head2 in
        let min_rank =
          match min_rank, Map.find t.bpe_ranks pair with
          | None, None -> None
          | None, Some value -> Some (value, pair)
          | Some (min_value, _), Some value when value < min_value -> Some (value, pair)
          | (Some _ as some), (None | Some _) -> some
        in
        best_pair words min_rank
    in
    match best_pair words None with
    | None -> words
    | Some (_min_value, (word1, word2)) ->
      let word12 = word1 ^ word2 in
      let rec group words acc =
        match words with
        | [] -> List.rev acc
        | [ last ] -> List.rev (last :: acc)
        | head1 :: head2 :: tail when String.(head1 = word1 && head2 = word2) ->
          group tail (word12 :: acc)
        | head :: tail -> group tail (head :: acc)
      in
      group words [] |> loop
  in
  unicode_chars str |> loop |> List.map ~f:(Map.find_exn t.encoder)

(* Run bpe on the whole string, very very inefficient but should be good enough for
   prompts. The original string should first be split on whitespace/... *)
let encode =
  let whitespace_pattern = String.Search_pattern.create " " in
  fun t str ->
    String.Search_pattern.replace_all whitespace_pattern ~in_:(" " ^ str) ~with_:delim
    |> bpe t

let decode t ids =
  List.map ids ~f:(fun i -> Map.find_exn t.decoder i) |> String.concat ~sep:""

let create ~config_filename =
  let to_assoc_exn ~key = function
    | None -> [%message "cannot find" key config_filename] |> failwith_s
    | Some (`Assoc v) -> v
    | Some _ -> [%message "unexpected type" key config_filename] |> failwith_s
  in
  let config = Yojson.Safe.from_file config_filename in
  let model =
    match config with
    | `Assoc assoc ->
      List.Assoc.find assoc "model" ~equal:String.equal
      |> to_assoc_exn ~key:"model"
      |> Map.of_alist_exn (module String)
    | _ -> [%message "json config is not an object" config_filename] |> failwith_s
  in
  let model_type = Map.find model "type" in
  if Caml.( <> ) model_type (Some (`String "BPE"))
  then [%message "unexpected model.type" config_filename] |> failwith_s;
  let vocab =
    Map.find model "vocab"
    |> to_assoc_exn ~key:"model.vocab"
    |> List.map ~f:(function
         | key, `Int i -> key, i
         | key, _ ->
           [%message "unexpected type in vocab" key config_filename] |> failwith_s)
  in
  let single_chars =
    List.filter_map vocab ~f:(fun (key, _) ->
      if String.length key = 1 then Some key else None)
    |> Set.of_list (module String)
  in
  let encoder =
    List.filter_map vocab ~f:(fun (key, value) ->
      let key =
        match String.chop_prefix key ~prefix:"<0x" with
        | None -> Some key
        | Some value ->
          (match String.chop_suffix value ~suffix:">" with
           | None -> Some key
           | Some value ->
             let value =
               Int.of_string ("0x" ^ value) |> Char.of_int_exn |> String.of_char
             in
             if Set.mem single_chars value then None else Some value)
      in
      Option.map key ~f:(fun key -> key, value))
  in
  let decoder =
    List.map encoder ~f:(fun (key, value) ->
      let key =
        unicode_chars key
        |> List.map ~f:(fun unicode_char ->
             if String.( = ) unicode_char delim then " " else unicode_char)
        |> String.concat ~sep:""
      in
      value, key)
  in
  let bpe_ranks =
    match Map.find model "merges" with
    | None -> [%message "cannot find model.merges" config_filename] |> failwith_s
    | Some (`List v) ->
      List.mapi v ~f:(fun i s ->
        match s with
        | `String s ->
          (match String.split s ~on:' ' with
           | [ s1; s2 ] -> (s1, s2), i
           | _ ->
             [%message "unexpected string in model.merges" s config_filename]
             |> failwith_s)
        | other ->
          let other = Yojson.Safe.to_string other in
          [%message "unexpected type in model.merges" other config_filename] |> failwith_s)
      |> Map.of_alist_exn (module String_pair)
    | Some _ -> [%message "unexpected type model.merges" config_filename] |> failwith_s
  in
  { encoder = Map.of_alist_exn (module String) encoder
  ; decoder = Map.of_alist_exn (module Int) decoder
  ; bpe_ranks
  }
