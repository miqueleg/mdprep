/*
 * Minimal "stored" (uncompressed) ZIP writer.
 *
 * The bundle only needs to carry two or three small text files to Colab, so a
 * whole compression library would be more moving parts than the job deserves.
 * The byte layout is exercised against Python's `zipfile` in
 * tests/test_web_bundle_zip.py, which is the thing that actually has to read
 * it on the other side.
 */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function dosDateTime(date) {
  // ZIP stores MS-DOS timestamps, which start in 1980 and have 2-second
  // resolution. Anything earlier would wrap, so clamp to the epoch.
  const year = Math.max(1980, date.getFullYear());
  return {
    time:
      (date.getHours() << 11) |
      (date.getMinutes() << 5) |
      (Math.floor(date.getSeconds() / 2) & 0x1f),
    date:
      ((year - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate(),
  };
}

class ByteSink {
  constructor() {
    this.chunks = [];
    this.length = 0;
  }

  raw(bytes) {
    this.chunks.push(bytes);
    this.length += bytes.length;
  }

  u16(value) {
    this.raw(new Uint8Array([value & 0xff, (value >>> 8) & 0xff]));
  }

  u32(value) {
    this.raw(
      new Uint8Array([
        value & 0xff,
        (value >>> 8) & 0xff,
        (value >>> 16) & 0xff,
        (value >>> 24) & 0xff,
      ]),
    );
  }

  concat() {
    const out = new Uint8Array(this.length);
    let offset = 0;
    for (const chunk of this.chunks) {
      out.set(chunk, offset);
      offset += chunk.length;
    }
    return out;
  }
}

/**
 * Build a ZIP archive from [name, textContent] pairs.
 * @returns {Uint8Array}
 */
function storedZipBytes(files, now = new Date()) {
  const encoder = new TextEncoder();
  const { time, date } = dosDateTime(now);
  const sink = new ByteSink();
  const entries = [];

  for (const [name, text] of files) {
    const nameBytes = encoder.encode(name);
    const data = encoder.encode(text);
    const crc = crc32(data);
    entries.push({ nameBytes, crc, size: data.length, offset: sink.length });

    sink.u32(0x04034b50); // local file header
    sink.u16(20); // version needed
    sink.u16(0x0800); // UTF-8 names
    sink.u16(0); // stored, no compression
    sink.u16(time);
    sink.u16(date);
    sink.u32(crc);
    sink.u32(data.length); // compressed size == uncompressed size
    sink.u32(data.length);
    sink.u16(nameBytes.length);
    sink.u16(0); // no extra field
    sink.raw(nameBytes);
    sink.raw(data);
  }

  const centralStart = sink.length;
  for (const entry of entries) {
    sink.u32(0x02014b50); // central directory header
    sink.u16(20); // version made by
    sink.u16(20); // version needed
    sink.u16(0x0800);
    sink.u16(0);
    sink.u16(time);
    sink.u16(date);
    sink.u32(entry.crc);
    sink.u32(entry.size);
    sink.u32(entry.size);
    sink.u16(entry.nameBytes.length);
    sink.u16(0); // extra
    sink.u16(0); // comment
    sink.u16(0); // disk number
    sink.u16(0); // internal attributes
    sink.u32(0o100644 << 16); // external attributes: regular file, rw-r--r--
    sink.u32(entry.offset);
    sink.raw(entry.nameBytes);
  }
  const centralSize = sink.length - centralStart;

  sink.u32(0x06054b50); // end of central directory
  sink.u16(0); // this disk
  sink.u16(0); // disk with central directory
  sink.u16(entries.length);
  sink.u16(entries.length);
  sink.u32(centralSize);
  sink.u32(centralStart);
  sink.u16(0); // no archive comment

  return sink.concat();
}

function makeStoredZip(files) {
  return new Blob([storedZipBytes(files)], { type: "application/zip" });
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { storedZipBytes, crc32 };
}
