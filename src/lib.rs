#![warn(clippy::pedantic)]
#![cfg_attr(not(feature = "std"), no_std)]
//! A crate for encoding and decoding data using [Bencode](https://en.wikipedia.org/wiki/Bencode) -
//! an encoding format most commonly used for torrent files.
#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(all(feature = "serde", not(feature = "std")))]
use alloc::format;
use core::{convert::TryFrom, num::TryFromIntError, str};
use itertools::Itertools;
pub use nom;
use nom::{
    branch::alt,
    bytes::complete::{tag, take, take_while1},
    combinator::{iterator, map, map_res, opt, recognize},
    error::ParseError,
    sequence::{delimited, pair, preceded},
    AsBytes, IResult,
};
use num_traits::{Num, PrimInt};
#[cfg(feature = "std")]
use std::{
    collections::{BTreeMap, HashMap, LinkedList, VecDeque},
    io::{self, Write},
};
#[cfg(not(feature = "std"))]
use {
    alloc::{
        collections::{BTreeMap, LinkedList, VecDeque},
        string::{String, ToString},
        vec::Vec,
    },
    hashbrown::HashMap,
};
#[cfg(feature = "serde")]
use {
    core::{fmt, marker::PhantomData},
    serde1::{
        de::{self, MapAccess, Visitor},
        ser::SerializeMap,
        Deserialize, Serialize, Serializer,
    },
};

#[cfg(not(feature = "std"))]
pub mod io {
    pub enum Error {}
    pub type Result<T> = core::result::Result<T, Error>;

    pub trait Write {
        /// This is no-std emulation of [`std::io::Write::write_all`].
        ///
        /// # Errors
        /// Not supported yet, actually
        fn write_all(&mut self, buf: &[u8]) -> Result<()>;
    }

    impl Write for alloc::vec::Vec<u8> {
        fn write_all(&mut self, buf: &[u8]) -> Result<()> {
            self.extend_from_slice(buf);
            Ok(())
        }
    }
}

#[cfg(not(feature = "std"))]
pub use io::Write;

/// A trait to be implemented by every type that can be encoded and decoded using
/// [Bencode](https://en.wikipedia.org/wiki/Bencode).
pub trait Bencodable<'a>: Sized + 'a {
    /// The error type used by the [bdecode](Bencodable::bdecode) parser.
    ///
    /// Use `nom::error::Error<&'a [u8]>` as the default (default associated types aren't supported yet).
    ///
    /// See also: [`nom::Err`].
    type Error: ParseError<&'a [u8]>;
    /// Deserialize from bencoded data.
    ///
    /// # Errors
    /// An error is returned if the data couldn't be parsed.
    ///
    /// # Example
    /// ```
    /// # use trebuchet_bencode::Bencodable;
    /// let input = b"i5ei6enon-bencoded data";
    /// // Parse a bencoded value, get leftover data and the value
    /// let (input, a) = i32::bdecode(input)?;
    /// // Parse one more value
    /// let (input, b) = i32::bdecode(input)?;
    /// assert_eq!(a, 5);
    /// assert_eq!(b, 6);
    /// assert_eq!(input, b"non-bencoded data");
    /// # Ok::<(), nom::Err<<i32 as Bencodable>::Error>>(())
    /// ```
    fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self, Self::Error>;
    /// Serialize into a stream using bencoding.
    ///
    /// # Errors
    /// An error is returned if the stream can't be written to due to an I/O error.
    ///
    /// # Example
    /// ```
    /// # use trebuchet_bencode::Bencodable;
    /// let mut serialized = Vec::<u8>::new();
    /// // Vec<u8> implements std::io::Write
    /// 1337.bencode(&mut serialized)?;
    /// assert_eq!(&serialized, b"i1337e");
    /// # Ok::<(), std::io::Error>(())
    /// ```
    fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()>;
}

fn decimal<T: PrimInt>(input: &[u8]) -> IResult<&[u8], T> {
    map_res(
        recognize(preceded(
            opt(tag("-")),
            take_while1(|x| matches!(x, b'0'..=b'9')),
        )),
        // Should be panic-free because it's guaranteed to consist entirely of -0123456789
        // (As long as the parser works properly)
        |s| Num::from_str_radix(str::from_utf8(s).unwrap(), 10),
    )(input)
}

fn bdecode_int<T: PrimInt>(input: &[u8]) -> IResult<&[u8], T> {
    delimited(tag(b"i"), decimal, tag(b"e"))(input)
}

macro_rules! int_impl {
    ($($t:ty)*) => {$(
        impl<'a> Bencodable<'a> for $t {
            type Error = nom::error::Error<&'a [u8]>;
            fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self> {
                bdecode_int(input)
            }
            fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
                stream.write_all(b"i")?;
                stream.write_all(self.to_string().as_ref())?;
                stream.write_all(b"e")
            }
        }
    )*};
}

// u8 isn't implemented not to conflict with Vec<u8> as owned byte array...
// not the best solution is it...
int_impl!(i8 i16 i32 i64 i128 isize u16 u32 u64 u128 usize);

impl<'a> Bencodable<'a> for &'a [u8] {
    type Error = nom::error::Error<&'a [u8]>;
    fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self> {
        let (input, len) = decimal::<usize>(input)?;
        preceded(tag(b":"), take(len))(input)
    }
    fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
        stream.write_all(self.len().to_string().as_ref())?;
        stream.write_all(b":")?;
        stream.write_all(self)
    }
}

impl<'a> Bencodable<'a> for Vec<u8> {
    type Error = nom::error::Error<&'a [u8]>;
    fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self> {
        map(<&'a [u8]>::bdecode, From::from)(input)
    }
    fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
        self.as_slice().bencode(stream)
    }
}

impl<'a> Bencodable<'a> for &'a str {
    type Error = nom::error::Error<&'a [u8]>;
    fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self> {
        let (input, len) = decimal::<usize>(input)?;
        preceded(tag(b":"), map_res(take(len), str::from_utf8))(input)
    }
    fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
        self.as_bytes().bencode(stream)
    }
}

impl<'a> Bencodable<'a> for String {
    type Error = nom::error::Error<&'a [u8]>;
    fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self> {
        map(map_res(<&'a [u8]>::bdecode, str::from_utf8), From::from)(input)
    }
    fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
        self.as_str().bencode(stream)
    }
}

/// Vec doesn't have `push_back` implemented, so implement it ourselves so we can
/// use the default From implementation.
trait VecFromHelper<T> {
    fn push_back(&mut self, x: T);
}

impl<T> VecFromHelper<T> for Vec<T> {
    fn push_back(&mut self, x: T) {
        self.push(x);
    }
}

macro_rules! impl_vec {
    ($($ty:ty)*) => {$(
        impl<'a, T: Bencodable<'a>> Bencodable<'a> for $ty {
            type Error = T::Error;
            fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self, Self::Error> {
                delimited(
                    tag(b"l"),
                    |input| {
                        let mut it = iterator(
                            input,
                            T::bdecode,
                        );
                        let ret = it.collect::<Self>();
                        Ok((it.finish()?.0, ret))
                    },
                    tag(b"e"),
                )(input)
            }
            fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
                stream.write_all(b"l")?;
                self.iter().try_for_each(|x| x.bencode(stream))?;
                stream.write_all(b"e")
            }
        }

        impl<'a, T: Into<BencodeAny<'a>>> From<$ty> for BencodeAny<'a> {
            fn from(x: $ty) -> Self {
                let mut ret = Vec::new();
                for v in x {
                    ret.push_back(v.into());
                }
                Self::List(ret)
            }
        }

        impl<T: Into<BencodeAnyOwned>> From<$ty> for BencodeAnyOwned {
            fn from(x: $ty) -> Self {
                let mut ret = Vec::new();
                for v in x {
                    ret.push_back(v.into());
                }
                Self::List(ret)
            }
        }
    )*};
}

impl_vec!(Vec<T> VecDeque<T> LinkedList<T>);

macro_rules! impl_hashmap {
    ($($ty:ty)*) => {$(
        impl<'a, T: Bencodable<'a>> Bencodable<'a> for $ty
        where
            T::Error: From<nom::error::Error<&'a [u8]>>,
        {
            type Error = T::Error;
            fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self, Self::Error> {
                delimited(
                    tag(b"d"),
                    |input| {
                        let mut it = iterator(
                            input,
                            pair(
                                |input| <&'a [u8]>::bdecode(input).map_err(nom::Err::convert),
                                T::bdecode,
                            ),
                        );
                        let ret = it.map(|(k, v)| (k.into(), v)).collect::<Self>();
                        Ok((it.finish()?.0, ret))
                    },
                    tag(b"e"),
                )(input)
            }
            fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
                stream.write_all(b"d")?;
                self.iter()
                    .sorted_by(|(a, _), (b, _)| Ord::cmp(a, b))
                    .try_for_each(|(k, v)| k.bencode(stream).and_then(|_| v.bencode(stream)))?;
                stream.write_all(b"e")
            }
        }
    )*};
}

macro_rules! impl_from_hashmap {
    ($($ty:ident)*) => {$(
        impl<'a, T> From<$ty<&'a [u8], T>> for BencodeAny<'a> where BencodeAny<'a>: From<T> {
            fn from(x: $ty<&'a [u8], T>) -> Self {
                let mut ret = HashMap::<&'a [u8], BencodeAny<'a>>::new();
                for (k, v) in x {
                    ret.insert(k, v.into());
                }
                BencodeAny::Dict(ret)
            }
        }

        impl<K: AsRef<[u8]>, T: Into<BencodeAnyOwned>> From<$ty<K, T>> for BencodeAnyOwned {
            fn from(x: $ty<K, T>) -> Self {
                let mut ret = HashMap::<Vec<u8>, BencodeAnyOwned>::new();
                for (k, v) in x {
                    ret.insert(k.as_ref().into(), v.into());
                }
                BencodeAnyOwned::Dict(ret)
            }
        }
    )*};
}

impl_hashmap!(HashMap<Vec<u8>, T> HashMap<&'a [u8], T> BTreeMap<Vec<u8>, T> BTreeMap<&'a [u8], T>);
impl_from_hashmap!(HashMap BTreeMap);

/// An enum representing any bencoded datatype with a lifetime.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BencodeAny<'a> {
    /// An integer.
    Int(i64),
    /// A list.
    List(Vec<BencodeAny<'a>>),
    /// A key:value dictionary.
    Dict(HashMap<&'a [u8], BencodeAny<'a>>),
    /// A byte string. Might specify a text value, however it isn't guaranteed to be UTF-8. For
    /// torrents, encoding is sometimes specified in the `encoding` field of the metainfo
    /// dictionary (i.e. the top-level dictionary of the torrent file). Many torrent clients don't
    /// specify the encoding and silently use UTF-8, so that can be considered the default.
    Str(&'a [u8]),
}

/// An enum representing any bencoded datatype (owned).
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BencodeAnyOwned {
    /// An integer.
    Int(i64),
    /// A list.
    List(Vec<BencodeAnyOwned>),
    /// A key:value dictionary.
    Dict(HashMap<Vec<u8>, BencodeAnyOwned>),
    /// A byte string. Might specify a text value, however it isn't guaranteed to be UTF-8. For
    /// torrents, encoding is sometimes specified in the `encoding` field of the metainfo
    /// dictionary (i.e. the top-level dictionary of the torrent file). Many torrent clients don't
    /// specify the encoding and silently use UTF-8, so that can be considered the default.
    Str(Vec<u8>),
}

impl<'a> From<&'a BencodeAny<'a>> for BencodeAnyOwned {
    fn from(val: &'a BencodeAny<'a>) -> Self {
        match val {
            BencodeAny::Int(n) => Self::Int(*n),
            BencodeAny::List(v) => Self::List(v.iter().map(From::from).collect()),
            BencodeAny::Dict(d) => Self::Dict(
                d.iter()
                    .map(|(&k, v)| (Vec::from(k), From::from(v)))
                    .collect(),
            ),
            BencodeAny::Str(s) => Self::Str(s.to_vec()),
        }
    }
}

impl<'a> From<BencodeAny<'a>> for BencodeAnyOwned {
    fn from(val: BencodeAny<'a>) -> Self {
        Self::from(&val)
    }
}

impl<'a> BencodeAny<'a> {
    /// Creates an owned object from a borrowed object.
    #[allow(clippy::must_use_candidate)]
    pub fn into_owned(&self) -> BencodeAnyOwned {
        From::from(self)
    }
}

#[cfg(feature = "serde")]
mod serde {
    use super::{
        de, fmt, str, BencodeAny, BencodeAnyOwned, Deserialize, HashMap, MapAccess, PhantomData,
        TryFrom, Visitor,
    };
    #[cfg(not(feature = "std"))]
    use super::{format, String, Vec};
    #[derive(Default)]
    struct BencodeAnyVisitor<'a> {
        _ph: PhantomData<&'a u8>,
    }

    #[derive(Default)]
    struct BencodeAnyOwnedVisitor;

    macro_rules! default_visitor_impl {
        () => {
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a bencoded value")
            }
            fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut ret: HashMap<_, Self::Value> =
                    HashMap::with_capacity(access.size_hint().unwrap_or(0));
                while let Some((k, v)) = access.next_entry::<&'de [u8], _>()? {
                    ret.insert(k, v);
                }
                Ok(ret.into())
            }
            fn visit_seq<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut ret: Vec<Self::Value> = Vec::with_capacity(access.size_hint().unwrap_or(0));
                while let Some(v) = access.next_element()? {
                    ret.push(v);
                }
                Ok(ret.into())
            }
            fn visit_newtype_struct<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: serde1::Deserializer<'de>,
            {
                deserializer.deserialize_any(Self::default())
            }
            fn visit_i128<E>(self, value: i128) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                TryFrom::try_from(value).map_err(|_| {
                    E::custom(format!("bencode integer can't fit into i64: {}", value))
                })
            }
            fn visit_u128<E>(self, value: u128) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                TryFrom::try_from(value).map_err(|_| {
                    E::custom(format!("bencode integer can't fit into i64: {}", value))
                })
            }
            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                TryFrom::try_from(value).map_err(|_| {
                    E::custom(format!("bencode integer can't fit into i64: {}", value))
                })
            }
            fn visit_u32<E>(self, value: u32) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(From::from(value))
            }
            fn visit_u16<E>(self, value: u16) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(From::from(value))
            }
            fn visit_u8<E>(self, value: u8) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(From::from(value as u16))
            }
            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(From::from(value))
            }
            fn visit_i32<E>(self, value: i32) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(From::from(value))
            }
            fn visit_i16<E>(self, value: i16) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(From::from(value))
            }
            fn visit_i8<E>(self, value: i8) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(From::from(value))
            }
        };
    }

    impl<'a, 'de: 'a> Visitor<'de> for BencodeAnyVisitor<'a> {
        type Value = BencodeAny<'a>;
        default_visitor_impl!();
        fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(From::from(v))
        }
        fn visit_borrowed_bytes<E>(self, v: &'de [u8]) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(From::from(v))
        }
    }

    impl<'de> Visitor<'de> for BencodeAnyOwnedVisitor {
        type Value = BencodeAnyOwned;
        default_visitor_impl!();
        fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(From::from(v))
        }
        fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(From::from(v))
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(From::from(v))
        }
        fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(From::from(v))
        }
    }

    impl<'a, 'de: 'a> Deserialize<'de> for BencodeAny<'a> {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde1::Deserializer<'de>,
        {
            deserializer.deserialize_any(BencodeAnyVisitor::<'a>::default())
        }
    }

    impl<'de> Deserialize<'de> for BencodeAnyOwned {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde1::Deserializer<'de>,
        {
            deserializer.deserialize_any(BencodeAnyOwnedVisitor::default())
        }
    }
}

macro_rules! impl_any {
    ($($ty:ty)*) => {$(
        impl<'a> Bencodable<'a> for $ty {
            type Error = nom::error::Error<&'a [u8]>;
            fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self> {
                alt((
                    map(Bencodable::bdecode, Self::Int),
                    map(Bencodable::bdecode, Self::List),
                    map(Bencodable::bdecode, Self::Dict),
                    map(Bencodable::bdecode, Self::Str),
                ))(input)
            }
            fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
                match self {
                    Self::Int(x) => x.bencode(stream),
                    Self::List(x) => x.bencode(stream),
                    Self::Dict(x) => x.bencode(stream),
                    Self::Str(x) => x.bencode(stream),
                }
            }
        }

        impl<'a> From<i64> for $ty { fn from(n: i64) -> Self { Self::Int(n.into()) } }
        impl<'a> From<i32> for $ty { fn from(n: i32) -> Self { Self::Int(n.into()) } }
        impl<'a> From<i16> for $ty { fn from(n: i16) -> Self { Self::Int(n.into()) } }
        impl<'a> From<i8> for $ty { fn from(n: i8) -> Self { Self::Int(n.into()) } }
        impl<'a> From<u32> for $ty { fn from(n: u32) -> Self { Self::Int(n.into()) } }
        impl<'a> From<u16> for $ty { fn from(n: u16) -> Self { Self::Int(n.into()) } }
        impl<'a> TryFrom<u128> for $ty {
            type Error = TryFromIntError;
            fn try_from(n: u128) -> Result<Self, TryFromIntError> { TryFrom::try_from(n).map(Self::Int) }
        }
        impl<'a> TryFrom<u64> for $ty {
            type Error = TryFromIntError;
            fn try_from(n: u64) -> Result<Self, TryFromIntError> { TryFrom::try_from(n).map(Self::Int) }
        }
        impl<'a> TryFrom<i128> for $ty {
            type Error = TryFromIntError;
            fn try_from(n: i128) -> Result<Self, TryFromIntError> { TryFrom::try_from(n).map(Self::Int) }

        }
        impl<'a> From<&'a str> for $ty { fn from(s: &'a str) -> Self { Self::Str(s.as_bytes().into()) } }
        impl<'a> From<&'a [u8]> for $ty { fn from(s: &'a [u8]) -> Self { Self::Str(s.into()) } }

        #[cfg(feature = "serde")]
        impl<'a> Serialize for $ty {
            fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
                where S: Serializer {
                match self {
                    Self::Int(x) => x.serialize(ser),
                    Self::List(x) => x.serialize(ser),
                    Self::Dict(x) => {
                        let mut map = ser.serialize_map(Some(x.len()))?;
                        for (k, v) in x {
                            map.serialize_entry(k as &[u8], v)?;
                        }
                        map.end()
                    },
                    Self::Str(x) => x.serialize(ser),
                }
            }
        }
    )*};
}

impl_any!(BencodeAny<'a> BencodeAnyOwned);

impl From<String> for BencodeAnyOwned {
    fn from(x: String) -> Self {
        Self::Str(x.into())
    }
}

impl From<Vec<u8>> for BencodeAnyOwned {
    fn from(x: Vec<u8>) -> Self {
        Self::Str(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;

    /// Strip lifetime from a u8 slice.
    fn s(x: &[u8]) -> &[u8] {
        x
    }

    fn verify1<'a, T: Debug + std::cmp::PartialEq + Bencodable<'a>>(
        test: &'a [u8],
        expected: &T,
        expected_back: Option<&'a [u8]>,
        leftover: Option<&'a [u8]>,
    ) where
        T::Error: Debug,
    {
        let (input, dec) = T::bdecode(test).unwrap();
        assert_eq!(&dec, expected);
        assert_eq!(input, leftover.unwrap_or(b""));
        let mut out = Vec::<u8>::new();
        dec.bencode(&mut out).unwrap();
        assert_eq!(out, expected_back.unwrap_or(test));
    }

    fn verify<'a, T: Debug + std::cmp::PartialEq + Bencodable<'a>>(test: &'a [u8], expected: &T)
    where
        T::Error: Debug,
    {
        verify1(test, expected, None, None)
    }

    fn ensure_fail<'a, T: Bencodable<'a>>(test: &'a [u8]) {
        assert!(T::bdecode(test).is_err());
    }

    #[test]
    fn int_values() {
        verify(b"i-6234e", &-6234);
        ensure_fail::<i64>(b"i-6234");
        ensure_fail::<u16>(b"i65536e");
        ensure_fail::<u64>(b"i-6234e");
        verify(b"i18446744073709551615e", &18446744073709551615u64);
        verify(b"i-9223372036854775808e", &-9223372036854775808i64);
    }

    #[test]
    fn string_values() {
        verify(b"4:abcd", &s(b"abcd"));
        verify(b"4:abcd", &vec![b'a', b'b', b'c', b'd']);
        ensure_fail::<Vec<u8>>(b"5:abcd");
        verify1(b"3:abcdef", &s(b"abc"), Some(b"3:abc"), Some(b"def"));
        verify(b"0:", &s(b""));
        ensure_fail::<Vec<u8>>(b":abcd");
    }

    #[test]
    fn list_values() {
        // Be careful not to use Vec<u8>, as that's the string type!
        verify1(
            b"li1ei2ei3eee",
            &vec![1, 2, 3],
            Some(b"li1ei2ei3ee"),
            Some(b"e"),
        );
        verify1(
            b"l4:spam4:eggsi5eei5e",
            &vec![
                BencodeAny::Str(b"spam"),
                BencodeAny::Str(b"eggs"),
                BencodeAny::Int(5),
            ],
            Some(b"l4:spam4:eggsi5ee"),
            Some(b"i5e"),
        );
        verify(b"le", &Vec::<u16>::new());
        ensure_fail::<Vec<u16>>(b"e");
        ensure_fail::<Vec<u16>>(b"li65536ei65536ee");
    }

    #[test]
    fn dict_values() {
        verify1(
            b"d4:testi1337e4:spaml1:a1:beee",
            &vec![
                (s(b"test"), BencodeAny::Int(1337)),
                (
                    s(b"spam"),
                    BencodeAny::List(vec![BencodeAny::Str(b"a"), BencodeAny::Str(b"b")]),
                ),
            ]
            .drain(..)
            .collect::<HashMap<&[u8], BencodeAny>>(),
            Some(b"d4:spaml1:a1:be4:testi1337ee"), // Serialization must be sorted
            Some(b"e"),
        );
        ensure_fail::<HashMap<&[u8], BencodeAny>>(b"d4:spaml1:a1:be4:testti1337ee");
        ensure_fail::<HashMap<&[u8], BencodeAny>>(b"l4:spaml1:a1:be4:testi1337ee");
        verify(b"de", &HashMap::<&[u8], u16>::new());
        ensure_fail::<HashMap<&[u8], BencodeAny>>(b"dd4:spaml1:a1:be4:testi1337ee");
    }
}
