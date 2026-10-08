const PASSWORD_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%^&*()-_=+";

function secureRandomIndex(maxExclusive, cryptoSource = globalThis.crypto) {
  if (!Number.isInteger(maxExclusive) || maxExclusive < 2 || maxExclusive > 256) {
    throw new Error("Password alphabet size must be between 2 and 256 characters.");
  }
  if (!cryptoSource?.getRandomValues) {
    throw new Error("Secure random number generation is unavailable.");
  }
  const limit = 256 - (256 % maxExclusive);
  const bytes = new Uint8Array(1);
  do {
    cryptoSource.getRandomValues(bytes);
  } while (bytes[0] >= limit);
  return bytes[0] % maxExclusive;
}

function generatePassword(length = 16, cryptoSource = globalThis.crypto) {
  if (!Number.isInteger(length) || length < 1) {
    throw new Error("Password length must be a positive integer.");
  }
  let password = "";
  for (let index = 0; index < length; index += 1) {
    password += PASSWORD_ALPHABET[secureRandomIndex(PASSWORD_ALPHABET.length, cryptoSource)];
  }
  return password;
}

if (typeof module !== "undefined") {
  module.exports = { PASSWORD_ALPHABET, generatePassword, secureRandomIndex };
}
