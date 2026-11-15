# Changelog

## 1.0.6

- `requires_hermes` corrected to `>=0.22.2`. 1.0.5 declared `>=0.23.0` on the assumption that the
  `register_tool` / `get_config` API arrived there; it did not — a 0.22.2 install has the same
  signature (`hermes_cli/plugins.py`, `PluginContext.register_tool` with `check_fn`, `emoji`,
  `override`) and `get_config`, so the higher floor only locked out working hosts.

## 1.0.5

- `files`: local documents are asked about, not inlined. Each file is uploaded to the DeepSeek
  Files API (`purpose` before `file`, `rich_layout_recognition` for PDF/Word/PowerPoint,
  `user_data` for spreadsheets, plain text and images) and referenced by `file_id` — the vendor
  refuses documents inside a chat request. Upload ids are cached by content hash, so the same
  document costs one upload.
- The upload name is sanitised (quotes, CR/LF and non-ASCII folded, extension preserved): a local
  file name can no longer truncate or inject a multipart header.
- `image_url` is validated against the documented contract (https or `data:image/`, ≤8192 chars)
  and ignored otherwise instead of turning a typo into a billed 400.
- The credential-path guard normalises Windows separators, so `C:\Users\x\.local\secrets\…` is
  refused like its POSIX spelling.
- Documentation corrected where it had drifted from the code: the plugin does write one file (the
  id cache, mode 0600, no path and no file name), uploads stay on the vendor's side, and a
  document passed in `files` is an injection channel like a web page.
- `requires_hermes` raised to `>=0.23.0`, the version the `register_tool` / `get_config`
  signatures were verified against.

## 1.0.4

- `files`: local documents (PDF, Office, text, images) can be attached to the question.

## 1.0.3

- `thinking` and `with_read_url` are independent switches; `model_name` is the config key.

## 1.0.2

- Retry on the transient status set, with the vendor's reason attached to each failure.

## 1.0.1

- `DEEPSEEK_BASE_URL` is validated (exact origin or loopback) instead of trusted.

## 1.0.0

- First release: `web_research` over DeepSeek's server-side web tools, with citations.
