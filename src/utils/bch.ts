const GF_SIZE = 256;
const GF_POLY = 0x011d;

class BCH {
  private poly: number;
  private msg_len: number;
  private ecc_len: number;
  private nsym: number;
  private alpha_to: Uint16Array;
  private index_of: Uint16Array;
  private gen_poly: Uint8Array;

  constructor(msg_bits = 56, ecc_bits = 44) {
    this.msg_len = Math.ceil(msg_bits / 8);
    this.ecc_len = Math.ceil(ecc_bits / 8);
    this.nsym = this.ecc_len;
    this.poly = 0x10d;

    this.alpha_to = new Uint16Array(GF_SIZE);
    this.index_of = new Uint16Array(GF_SIZE);
    this._init_gf();

    this.gen_poly = this._build_gen_poly();
  }

  private _init_gf() {
    let mask = 1;
    for (let i = 0; i < GF_SIZE; i++) {
      this.alpha_to[i] = mask;
      this.index_of[mask] = i;
      mask <<= 1;
      if (mask & GF_SIZE) mask ^= this.poly;
    }
    this.index_of[0] = -1;
  }

  private _build_gen_poly(): Uint8Array {
    const poly = new Uint8Array(this.nsym + 1);
    poly[0] = 1;
    for (let i = 1; i <= this.nsym; i++) {
      for (let j = i; j > 0; j--) {
        poly[j] = poly[j - 1] ^ this._gf_mul(poly[j], this.alpha_to[i]);
      }
      poly[0] = this._gf_mul(poly[0], this.alpha_to[i]);
    }
    return poly;
  }

  private _gf_mul(a: number, b: number): number {
    if (a === 0 || b === 0) return 0;
    return this.alpha_to[(this.index_of[a] + this.index_of[b]) % (GF_SIZE - 1)];
  }

  encode(data: Uint8Array): Uint8Array {
    const msg_in = new Uint8Array(data.length + this.nsym);
    msg_in.set(data);

    for (let i = 0; i < data.length; i++) {
      let feedback = msg_in[i] ^ msg_in[data.length + i];
      if (feedback === 0) continue;
      for (let j = 1; j < this.nsym; j++) {
        msg_in[data.length + j - 1] ^= this._gf_mul(
          this.gen_poly[this.nsym - j],
          feedback
        );
      }
    }

    const result = new Uint8Array(data.length + this.nsym);
    result.set(data);
    result.set(msg_in.slice(data.length), data.length);
    return result;
  }

  decode(data: Uint8Array): Uint8Array {
    const msg_in = new Uint8Array(data);
    const syndrome = new Uint8Array(this.nsym);

    for (let i = 0; i < this.nsym; i++) {
      for (let j = 0; j < data.length; j++) {
        syndrome[i] ^= this._gf_mul(msg_in[j], this.alpha_to[i]);
      }
    }

    let err_pos = 0;
    for (let i = 0; i < this.nsym; i++) {
      if (syndrome[i] !== 0) {
        err_pos |= 1 << i;
      }
    }

    if (err_pos === 0) return msg_in.slice(0, this.msg_len);
    if (err_pos >= (1 << this.nsym)) {
      return msg_in.slice(0, this.msg_len);
    }

    const result = new Uint8Array(msg_in);
    result[err_pos >> 3] ^= 1 << (err_pos & 7);
    return result.slice(0, this.msg_len);
  }
}

export default BCH;
