/**
 * Wire types for the TokenMiner admin API.
 *
 * Mirrors the records in TokenMiner.Contracts. ASP.NET serialises with the default camelCase
 * policy, so the field names here match the JSON on the wire rather than the C# properties.
 */

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

export interface PendingActivation {
  requiresActivation: boolean;
}

export interface UserProfile {
  id: string;
  email: string;
  displayName: string | null;
  status: string;
  emailConfirmed: boolean;
  roles: string[];
  createdAt: string;
}

// --- Mining catalogue -----------------------------------------------------------------------

export interface Coin {
  id: string;
  code: string;
  name: string;
  network: string;
  decimals: number;
  lastKnownUsdValue: number | null;
  lastValueDate: string | null;
  status: string;
  createdAt: string;
  updatedAt: string;
}

export interface PoolDetails {
  baseUrl: string;
  statusEndpoint: string | null;
  stratumEndpoint: string | null;
  coinId: string;
  payoutMode: string;
  payoutAddress: string | null;
  payoutNetwork: string | null;
}

export interface Pool {
  id: string;
  systemPoolId: string;
  name: string;
  provider: string;
  status: string;
  details: PoolDetails | null;
  coinIds: string[];
  createdAt: string;
  updatedAt: string;
}

export interface MiningAlgo {
  id: string;
  code: string;
  name: string;
  status: string;
  priority: number;
  configuration: unknown;
  createdAt: string;
  updatedAt: string;
}

// --- Treasury -------------------------------------------------------------------------------

export interface PoolPayout {
  id: string;
  poolId: string;
  coinId: string;
  walletAddress: string | null;
  amount: number;
  redeemedAmount: number;
  transactionHash: string | null;
  requestedAt: string | null;
  receivedAt: string | null;
  confirmationsCount: number;
  status: string;
  createdAt: string;
  updatedAt: string;
}

export interface ConversionProvider {
  id: string;
  name: string;
  baseUrl: string;
  status: string;
  priority: number;
  credentialRef: string | null;
  supportedFeatures: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface ConversionRoute {
  id: string;
  conversionProviderId: string;
  conversionProviderName: string;
  sourceCoinId: string;
  sourceCoinCode: string;
  destinationCoinId: string;
  destinationCoinCode: string;
  sourceNetwork: string | null;
  destinationNetwork: string | null;
  enabled: boolean;
  priority: number;
  minimumAmount: number;
  createdAt: string;
  updatedAt: string;
}

export interface ConversionTransaction {
  id: string;
  conversionProviderId: string;
  conversionProviderName: string;
  sourceCoinCode: string;
  sourceAmount: number;
  destinationCoinCode: string;
  destinationAmount: number | null;
  exchangeRate: number | null;
  fees: number | null;
  sourceTransactionId: string | null;
  destinationTransactionId: string | null;
  status: string;
  idempotencyKey: string;
  error: string | null;
  createdAt: string;
  updatedAt: string;
  completedAt: string | null;
}

// --- LLM provider ---------------------------------------------------------------------------

export interface LlmProviderDepositAccount {
  id: string;
  coinId: string;
  coinCode: string;
  network: string;
  depositAddress: string;
  status: string;
}

export interface LlmProvider {
  id: string;
  name: string;
  endpoint: string;
  credentialRef: string | null;
  status: string;
  balance: number | null;
  reservedBalance: number | null;
  totalBalance: number | null;
  balanceCheckedAt: string | null;
  depositAccounts: LlmProviderDepositAccount[];
  createdAt: string;
  updatedAt: string;
}

export interface LlmProviderModel {
  id: string;
  modelId: string;
  name: string | null;
  inputCost: number | null;
  outputCost: number | null;
  cachedInputCost: number | null;
  currency: string;
  contextLength: number | null;
  capabilities: string | null;
  status: string;
  lastSyncedAt: string;
}

export interface ProviderDeposit {
  id: string;
  llmProviderId: string;
  coinId: string;
  network: string;
  address: string;
  amount: number;
  transactionHash: string | null;
  confirmations: number;
  status: string;
  providerCreditBefore: number | null;
  providerCreditAfter: number | null;
  idempotencyKey: string;
  error: string | null;
  createdAt: string;
  updatedAt: string;
  confirmedAt: string | null;
}

// --- System ---------------------------------------------------------------------------------

export interface SystemStatus {
  runningSessions: number;
  pausedSessions: number;
  pendingShares: number;
  conversionsInFlight: number;
  providerDepositsInFlight: number;
  providerBalance: number | null;
  providerReservedBalance: number | null;
  providerBalanceCheckedAt: string | null;
  capturedAt: string;
}

export interface SystemConfiguration {
  key: string;
  value: string;
  updatedAt: string;
}
