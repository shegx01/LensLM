// Generates a 1024x1024 solid-color PNG used as the source for `tauri icon`.
// Behaviorless placeholder: no external image deps, pure node:zlib + Buffer.
import { deflateSync } from 'node:zlib';
import { writeFileSync } from 'node:fs';

const size = 1024;
const [r, g, b, a] = [37, 99, 235, 255]; // LensLM placeholder blue

const crcTable = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const crcBuf = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crcBuf]);
}

const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(size, 0);
ihdr.writeUInt32BE(size, 4);
ihdr[8] = 8; // bit depth
ihdr[9] = 6; // color type: RGBA

const row = Buffer.alloc(1 + size * 4);
for (let x = 0; x < size; x++) {
  row[1 + x * 4] = r;
  row[2 + x * 4] = g;
  row[3 + x * 4] = b;
  row[4 + x * 4] = a;
}
const raw = Buffer.concat(Array.from({ length: size }, () => row));

const png = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  chunk('IHDR', ihdr),
  chunk('IDAT', deflateSync(raw)),
  chunk('IEND', Buffer.alloc(0))
]);

writeFileSync('app-icon.png', png);
console.log(`wrote app-icon.png (${png.length} bytes)`);
