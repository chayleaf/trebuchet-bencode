use itertools::Itertools;
use nom::{
    branch::alt,
    bytes::complete::{tag, take, take_while1},
    combinator::{iterator, map, map_res, opt, recognize},
    error::ParseError,
    multi::many0,
    sequence::{delimited, pair, preceded},
    IResult,
};
use num_traits::{Num, PrimInt};
use std::{
    collections::HashMap,
    io::{self, Write},
    str,
};

pub trait Bdecodable<'a>: Sized + 'a {
    type Error: ParseError<&'a [u8]>;
    fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self, Self::Error>;
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
    preceded(tag(b"i"), decimal)(input)
}

macro_rules! int_impl {
    ($($t:ty)*) => {$(
        impl<'a> Bdecodable<'a> for $t {
            type Error = nom::error::Error<&'a [u8]>;
            fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self> {
                bdecode_int(input)
            }
            fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
                stream.write_all(self.to_string().as_ref())
            }
        }
    )*}
}

// u8 isn't implemented not to conflict with Vec<u8> as owned byte array...
// not the best solution is it...
int_impl!(i8 i16 i32 i64 i128 isize u16 u32 u64 u128 usize);

impl<'a> Bdecodable<'a> for &'a [u8] {
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

impl<'a> Bdecodable<'a> for Vec<u8> {
    type Error = nom::error::Error<&'a [u8]>;
    fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self> {
        map(<&'a [u8]>::bdecode, From::from)(input)
    }
    fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
        self.as_slice().bencode(stream)
    }
}

impl<'a, T: Bdecodable<'a>> Bdecodable<'a> for Vec<T> {
    type Error = T::Error;
    fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self, Self::Error> {
        delimited(tag(b"l"), many0(T::bdecode), tag(b"e"))(input)
    }
    fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
        stream.write_all(b"l")?;
        self.iter().try_for_each(|x| x.bencode(stream))?;
        stream.write_all(b"e")
    }
}

impl<'a, T: Bdecodable<'a>> Bdecodable<'a> for HashMap<&'a [u8], T>
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
                let ret = it.collect::<Self>();
                let (input, _) = it.finish()?;
                Ok((input, ret))
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

impl<'a, T: Bdecodable<'a>> Bdecodable<'a> for HashMap<Vec<u8>, T>
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
                let ret = it.map(|(k, v)| (k.to_owned(), v)).collect::<Self>();
                let (input, _) = it.finish()?;
                Ok((input, ret))
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

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BdecodeAny<'a> {
    Int(i64),
    List(Vec<BdecodeAny<'a>>),
    Dict(HashMap<&'a [u8], BdecodeAny<'a>>),
    Str(&'a [u8]),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BdecodeAnyOwned {
    Int(i64),
    List(Vec<BdecodeAnyOwned>),
    Dict(HashMap<Vec<u8>, BdecodeAnyOwned>),
    Str(Vec<u8>),
}

impl<'a> From<&'a BdecodeAny<'a>> for BdecodeAnyOwned {
    fn from(val: &'a BdecodeAny<'a>) -> Self {
        match val {
            BdecodeAny::Int(n) => Self::Int(*n),
            BdecodeAny::List(v) => Self::List(v.iter().map(From::from).collect()),
            BdecodeAny::Dict(d) => Self::Dict(
                d.iter()
                    .map(|(&k, v)| (k.to_owned(), From::from(v)))
                    .collect(),
            ),
            BdecodeAny::Str(s) => Self::Str(s.to_vec()),
        }
    }
}

impl<'a> From<BdecodeAny<'a>> for BdecodeAnyOwned {
    fn from(val: BdecodeAny<'a>) -> Self {
        Self::from(&val)
    }
}

impl<'a> BdecodeAny<'a> {
    pub fn into_owned(&self) -> BdecodeAnyOwned {
        From::from(self)
    }
}

impl<'a> Bdecodable<'a> for BdecodeAny<'a> {
    type Error = nom::error::Error<&'a [u8]>;
    fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self> {
        alt((
            map(i64::bdecode, Self::Int),
            map(Vec::<BdecodeAny>::bdecode, Self::List),
            map(HashMap::<&[u8], BdecodeAny>::bdecode, Self::Dict),
            map(<&'a [u8]>::bdecode, Self::Str),
        ))(input)
    }
    fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
        match self {
            Self::Int(n) => n.bencode(stream),
            Self::List(v) => v.bencode(stream),
            Self::Dict(d) => d.bencode(stream),
            Self::Str(s) => s.bencode(stream),
        }
    }
}

impl<'a> Bdecodable<'a> for BdecodeAnyOwned {
    type Error = nom::error::Error<&'a [u8]>;
    fn bdecode(input: &'a [u8]) -> IResult<&'a [u8], Self> {
        alt((
            map(i64::bdecode, Self::Int),
            map(Vec::<BdecodeAnyOwned>::bdecode, Self::List),
            map(HashMap::<Vec<u8>, BdecodeAnyOwned>::bdecode, Self::Dict),
            map(Vec::<u8>::bdecode, Self::Str),
        ))(input)
    }
    fn bencode<W: Write>(&self, stream: &mut W) -> io::Result<()> {
        match self {
            Self::Int(n) => n.bencode(stream),
            Self::List(v) => v.bencode(stream),
            Self::Dict(d) => d.bencode(stream),
            Self::Str(s) => s.bencode(stream),
        }
    }
}

// TODO!
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
