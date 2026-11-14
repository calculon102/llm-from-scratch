# Sample corpora

Small, self-contained text collections used by examples, tests and the
tokenizer UI. Everything here is **original text written for this
repository** and released under CC0-1.0 (public domain dedication), so it
can be copied, modified and redistributed without attribution.

## `sample_de/` — noisy German-first corpus (T-201)

The point of this corpus is not literary quality; it is to exercise the
ingestion pipeline's failure modes in miniature:

| File | Encoding / endings | Noise it demonstrates |
| --- | --- | --- |
| `sprache_und_schrift.txt` | UTF-8, LF | clean prose, umlauts, `ß`, long compounds |
| `stadtnotizen.txt` | UTF-8, LF | duplicate boilerplate line, blank lines, a URL, a one-word page number |
| `archiv_cp1252.txt` | Windows-1252, CRLF | legacy encoding, German quotation marks `„…“` |
| `rechtschreibung_bom.txt` | UTF-8 with BOM, CRLF | BOM and Windows line endings |

Run ingestion on it with:

```sh
.venv/bin/python scripts/tokenizer.py --config configs/tokenizer_tiny_debug.yaml
```

The run reports which encoding each file used and how many lines were
dropped as blank/short/duplicate; see `corpus/report.json` in the run
folder. Expected: all four files decode and the duplicate line in
`stadtnotizen.txt` is dropped. `Seite 14` is only 8 characters, so it
survives the default `min_line_chars: 2`; raise the threshold (e.g. to
12) to watch the short-line filter remove it.
