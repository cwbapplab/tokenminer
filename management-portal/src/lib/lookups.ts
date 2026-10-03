import { useCallback, useEffect, useState } from 'react';
import { api } from './api';
import type { Coin, ConversionProvider, LlmProvider } from './types';

/**
 * Reference data that forms need in order to offer sensible choices (which coin, which
 * provider). Loaded once and shared, because almost every resource screen needs some of it.
 */
export interface Lookups {
  coins: Coin[];
  conversionProviders: ConversionProvider[];
  llmProviders: LlmProvider[];
  loading: boolean;
  error: string | null;
  reload: () => void;
}

export function useLookups(): Lookups {
  const [coins, setCoins] = useState<Coin[]>([]);
  const [conversionProviders, setConversionProviders] = useState<ConversionProvider[]>([]);
  const [llmProviders, setLlmProviders] = useState<LlmProvider[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  const reload = useCallback(() => setNonce((value) => value + 1), []);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);

      try {
        const [coinList, providerList, llmList] = await Promise.all([
          api.get<Coin[]>('/api/admin/coins'),
          api.get<ConversionProvider[]>('/api/admin/conversion-providers'),
          api.get<LlmProvider[]>('/api/admin/llm-providers'),
        ]);

        if (!cancelled) {
          setCoins(coinList);
          setConversionProviders(providerList);
          setLlmProviders(llmList);
        }
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : 'Could not load reference data.');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void load();

    return () => {
      cancelled = true;
    };
  }, [nonce]);

  return { coins, conversionProviders, llmProviders, loading, error, reload };
}

/** Convenience for the many places that only need id -> code lookups. */
export function coinCode(coins: Coin[], id: unknown): string {
  return coins.find((coin) => coin.id === id)?.code ?? '—';
}
