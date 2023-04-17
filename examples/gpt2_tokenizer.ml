(* vocab file: https://cdn.huggingface.co/distilgpt2-vocab.json
   merge file: https://cdn.huggingface.co/distilgpt2-merges.txt
*)
(* TODO: Unicode is not properly supported here. *)
open! Base

module Vocab = struct
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

  type t =
    { tokens : string array
    ; token_indices : (string, int, String.comparator_witness) Map.t
    ; bpe_ranks : (String_pair.t, int, String_pair.comparator_witness) Map.t
    ; cache : (string, int list) Hashtbl.t
    }

  let load ~vocab_filename ~merge_filename =
    let err ~kind =
      Printf.failwithf
        "error reading json file %s, expected string got %s"
        vocab_filename
        kind
        ()
    in
    let token_indices =
      match Yojson.Basic.from_file vocab_filename with
      | `Assoc assoc ->
        List.map assoc ~f:(fun (name, id) -> name, Yojson.Basic.Util.to_int id)
        |> Map.of_alist_exn (module String)
      | `Bool _ -> err ~kind:"bool"
      | `Float _ -> err ~kind:"float"
      | `Int _ -> err ~kind:"int"
      | `List _ -> err ~kind:"list"
      | `Null -> err ~kind:"null"
      | `String _ -> err ~kind:"string"
    in
    let len =
      1 + Map.fold token_indices ~init:0 ~f:(fun ~key:_ ~data acc -> Int.max acc data)
    in
    let tokens = Array.create ~len "" in
    Map.iteri token_indices ~f:(fun ~key ~data -> tokens.(data) <- key);
    let bpe_ranks =
      Stdio.In_channel.read_lines merge_filename
      |> List.tl_exn
      |> List.mapi ~f:(fun idx line ->
           String.strip line
           |> String.split ~on:' '
           |> function
           | [ str1; str2 ] -> (str1, str2), idx
           | _ -> Printf.failwithf "multiple space characters in line %d: %s" idx line ())
      |> Map.of_alist_exn (module String_pair)
    in
    { tokens; token_indices; bpe_ranks; cache = Hashtbl.create (module String) }

  let token_id t token = Map.find t.token_indices token
  let token t id = t.tokens.(id)

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

  let bytes_encoding str =
    let buf = Buffer.create 128 in
    String.iter str ~f:(fun chr ->
      let c = Char.to_int chr in
      if c <= 32
      then Uutf.Buffer.add_utf_8 buf (Uchar.of_scalar_exn (c + 256))
      else if 127 <= c && c <= 160
      then Uutf.Buffer.add_utf_8 buf (Uchar.of_scalar_exn (c + 162))
      else if c = 173
      then Uutf.Buffer.add_utf_8 buf (Uchar.of_scalar_exn 323)
      else Buffer.add_char buf chr);
    Buffer.contents buf |> unicode_chars

  let bpe_rank t pair = Map.find t.bpe_ranks pair

  (* byte-level Byte-Pair-Encoding *)
  let bpe_ t str =
    let rec loop words =
      let rec best_pair words min_rank =
        match words with
        | [] | [ _ ] -> min_rank
        | head1 :: (head2 :: _ as words) ->
          let pair = head1, head2 in
          let min_rank =
            match min_rank, bpe_rank t pair with
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
    loop (bytes_encoding str)

  let bpe t str =
    Hashtbl.find_or_add t.cache str ~default:(fun () ->
      bpe_ t str |> List.filter_map ~f:(token_id t))
end

type t =
  { vocab : Vocab.t
  ; pattern_re : Re.re
  ; lower_case : bool
  }

let pattern_re = {|'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\d+| ?[^\s\[a-zA-Z]\d]+|\s+|}

let create ~vocab_filename ~merge_filename =
  let vocab = Vocab.load ~vocab_filename ~merge_filename in
  let pattern_re = Re.Perl.re pattern_re in
  let pattern_re = Re.compile pattern_re in
  { vocab; pattern_re; lower_case = false }

let encode t str =
  Re.matches t.pattern_re str
  |> List.concat_map ~f:(fun str ->
       let str = if t.lower_case then String.lowercase str else str in
       Vocab.bpe t.vocab str)

let decode t ids = List.map ids ~f:(Vocab.token t.vocab) |> String.concat ~sep:""
