namespace TokenMiner.Contracts.System;

public sealed record SystemStatusResponse(
    int RunningSessions,
    int IdleSessions,
    int PendingShares,
    int ConversionsInFlight,
    int ProviderDepositsInFlight,
    decimal? ProviderBalance,
    decimal? ProviderReservedBalance,
    DateTimeOffset? ProviderBalanceCheckedAt,
    DateTimeOffset CapturedAt);

public sealed record SystemConfigurationResponse(string Key, string Value, DateTimeOffset UpdatedAt);

/// <summary><paramref name="Value"/> is a JSON literal, e.g. <c>5.0</c> or <c>true</c>.</summary>
public sealed record SetSystemConfigurationRequest(string Value);
