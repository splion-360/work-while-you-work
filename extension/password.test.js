const assert = require("node:assert/strict");
const test = require("node:test");
const { PASSWORD_ALPHABET, generatePassword, secureRandomIndex } = require("./password.js");

function deterministicCrypto(values) {
  let index = 0;
  return {
    getRandomValues(bytes) {
      bytes[0] = values[index % values.length];
      index += 1;
      return bytes;
    }
  };
}

test("generates a 16-character password from the allowed alphabet", () => {
  const password = generatePassword(16, deterministicCrypto([...Array(16).keys()]));
  assert.equal(password.length, 16);
  for (const character of password) {
    assert.ok(PASSWORD_ALPHABET.includes(character));
  }
});

test("uses rejection sampling to avoid modulo bias", () => {
  const crypto = deterministicCrypto([255, 0]);
  assert.equal(secureRandomIndex(PASSWORD_ALPHABET.length, crypto), 0);
});

test("rejects invalid lengths", () => {
  assert.throws(() => generatePassword(0, deterministicCrypto([0])), /positive integer/);
});
