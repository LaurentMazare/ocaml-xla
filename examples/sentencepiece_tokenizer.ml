(* Very naive implementation of the sentencepiece tokenizer. *)
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

let delim = "▁"

type t =
  { encoder : (string, int, String.comparator_witness) Map.t
  ; decoder : string array
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

let decode t ids = List.map ids ~f:(fun i -> t.decoder.(i)) |> String.concat ~sep:""
let create ~config_filename:_ = failwith "TODO"
