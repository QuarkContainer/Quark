use core::cmp::Ord;
use alloc::vec::Vec;
use core::ops::Deref;

pub struct SortArr<K: Ord + Sized + Clone + Copy, V:  Sized + Clone + Copy>(pub Vec<(K, V)>);

impl <K: Ord + Sized + Clone + Copy, V:  Sized + Clone + Copy> SortArr <K, V> {
    pub fn New(arr: &[(K, V)]) -> Self {
        let mut vec = arr.to_vec();
        vec.sort_by_key(|a| a.0);
        return Self(vec)
    }

    pub fn Get(&self, key: K) -> Option<V> {
        match self.0.binary_search_by_key(&key, |&(key, _val)| key) {
            Ok(idx) => return Some(self.0[idx].1),
            Err(_) => return None
        }
    }
}

impl <K: Ord + Sized + Clone + Copy, V:  Sized + Clone + Copy> Deref for SortArr <K, V> {
    type Target = Vec<(K, V)>;

    fn deref(&self) -> &Vec<(K, V)> {
        &self.0
    }
}