/**
 * Read the wignerf document out of an exported mp4's `comment` metadata tag
 * (core/videoexport.py writes it with `-metadata comment=<json>`), so a kept
 * video is self-restoring — no separate setup file to lose.
 *
 * No mp4 box parsing: the tag is plain ASCII JSON inside the file, and
 * `-movflags +faststart` puts moov (hence the tag) near the HEAD — measured
 * at byte 3581 of a 554 KB export. Scanning a slice for the marker is
 * shorter and more robust than walking moov→udta→meta→ilst, and the tail is
 * tried too in case a file was written without faststart.
 */

const MARKER = '{"generator":"wignerf"'
const SLICE = 4*1024*1024

/** The document, or null when the file carries no wignerf metadata. */
export async function extractWignerfDoc(file: File): Promise<unknown | null> {
  const head = await readSlice(file, 0, Math.min(SLICE, file.size))
  return scan(head) ?? (file.size > SLICE
    ? scan(await readSlice(file, file.size - SLICE, file.size))
    : null)
}

async function readSlice(file: File, from: number, to: number) {
  const buf = await file.slice(from, to).arrayBuffer()
  // fatal: false — the slice is mostly h264 payload; stray bytes become
  // U+FFFD, which the brace scanner below simply steps over
  return new TextDecoder('utf-8', { fatal: false }).decode(buf)
}

function scan(text: string): unknown | null {
  const start = text.indexOf(MARKER)
  if (start < 0) return null
  // brace matching, quote- and escape-aware: the tag is followed by binary
  // data, so the object's end must be found structurally, not by a search
  let depth = 0, inStr = false, esc = false
  for (let i = start; i < text.length; i++) {
    const ch = text[i]
    if (esc) { esc = false; continue }
    if (inStr) {
      if (ch === '\\') esc = true
      else if (ch === '"') inStr = false
      continue
    }
    if (ch === '"') inStr = true
    else if (ch === '{') depth++
    else if (ch === '}' && --depth === 0) {
      try {
        return JSON.parse(text.slice(start, i + 1))
      } catch {
        return null      // truncated by the slice boundary, or not ours
      }
    }
  }
  return null
}
