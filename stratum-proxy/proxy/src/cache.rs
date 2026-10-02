use std::collections::HashMap;
use std::hash::Hash;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

/// A minimal time-to-live cache.
///
/// The working set here is tiny (one routing table and one entry per recently seen worker), so
/// a plain map with per-entry expiry is enough; no eviction policy is needed.
pub struct TtlCache<K, V> {
    ttl: Duration,
    entries: RwLock<HashMap<K, Entry<V>>>,
}

struct Entry<V> {
    value: V,
    expires_at: Instant,
}

impl<K, V> TtlCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Returns the cached value unless it has expired.
    pub async fn get(&self, key: &K) -> Option<V> {
        let entries = self.entries.read().await;
        let entry = entries.get(key)?;

        if entry.expires_at <= Instant::now() {
            return None;
        }

        Some(entry.value.clone())
    }

    pub async fn insert(&self, key: K, value: V) {
        let mut entries = self.entries.write().await;

        // Opportunistic pruning keeps the map bounded without a background task.
        let now = Instant::now();
        entries.retain(|_, entry| entry.expires_at > now);

        entries.insert(
            key,
            Entry {
                value,
                expires_at: now + self.ttl,
            },
        );
    }

    #[cfg(test)]
    async fn len(&self) -> usize {
        self.entries.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_a_value_before_it_expires() {
        let cache: TtlCache<String, u32> = TtlCache::new(Duration::from_secs(60));
        cache.insert("a".to_string(), 1).await;

        assert_eq!(cache.get(&"a".to_string()).await, Some(1));
    }

    #[tokio::test]
    async fn misses_unknown_keys() {
        let cache: TtlCache<String, u32> = TtlCache::new(Duration::from_secs(60));

        assert_eq!(cache.get(&"missing".to_string()).await, None);
    }

    #[tokio::test]
    async fn expires_entries_past_their_ttl() {
        let cache: TtlCache<String, u32> = TtlCache::new(Duration::from_millis(1));
        cache.insert("a".to_string(), 1).await;

        tokio::time::sleep(Duration::from_millis(20)).await;

        assert_eq!(cache.get(&"a".to_string()).await, None);
    }

    #[tokio::test]
    async fn overwrites_existing_entries() {
        let cache: TtlCache<String, u32> = TtlCache::new(Duration::from_secs(60));
        cache.insert("a".to_string(), 1).await;
        cache.insert("a".to_string(), 2).await;

        assert_eq!(cache.get(&"a".to_string()).await, Some(2));
        assert_eq!(cache.len().await, 1);
    }
}
