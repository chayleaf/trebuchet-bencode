# Trebuchet Bencode

A crate for parsing [Bencode](https://en.wikipedia.org/wiki/Bencode) made with [nom](https://docs.rs/nom)

Nearly-zero-alloc deserializing is possible - containers like Vec must still allocate, but byte strings and dictionary keys won't be copied. When serializing dictionaries, keys are automatically sorted (not sorting keys would go against the standard), so input might not always match the output.

Serde support is feature-flagged - just enable the `serde` feature.

no_std support (with `alloc`) is there too, just disable the `std` feature and enable the `nostd` feature!

Currently supported types (`T` stands for any supported type):
- `Vec<T>` and `&[T]`
- `Vec<u8>` and '&[u8]` - a byte string
- `String` and `&str` - a UTF-8 string. **NOTE: STRINGS IN TORRENT FILES ARE NOT GUARANTEED TO BE UTF-8! ASSUMING STRINGS ARE UTF-8 WILL FAIL ON SOME TORRENT FILES.** You should only use this type if you're sure you won't have to deal with non-unicode data (e.g. if only serializing, or only handling your own data, or simply if you don't expect to find any edge cases in the wild).
- All integer primitives (Except `u8`, as that's used for strings in this crate. The alternative would be creating a newtype for `Vec<u8>` and '&[u8]` to represent byte strings, and I feel like that would've been less convenient).
- `VecDeque<T>`
- `LinkedList<T>`
- `HashMap<&[u8], T>` - borrowed keys
- `HashMap<Vec<u8>, T>` - owned
- `BTreeMap<T>`
- `BencodeAnyOwned` and `BencodeAny<'a>` - for storing any data (that is, integers up to 64 bits, lists, dictionaries or byte strings).
- ...and you can implement Bencode for custom data types, and you can even use custom parse errors.

An example of a type that isn't implemented on purpose is `HashSet<T>` - it doesn't have ordering, but Bencode lists are always ordered.

