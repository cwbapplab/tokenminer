namespace TokenMiner.Contracts.Admin;

/// <summary>
/// One row of the operator's user list. <paramref name="Id"/> is the user account id, which the
/// portal surfaces as the client id.
/// </summary>
/// <remarks>
/// Share counts cover the terminal reward states only: <paramref name="AcceptedShares"/> counts
/// accepted and already redeemed shares, <paramref name="RejectedShares"/> counts terminal reward
/// failures. In-flight (pending or processing) shares are not counted.
/// </remarks>
public sealed record AdminUserSummaryResponse(
    Guid Id,
    string Email,
    string? DisplayName,
    string Status,
    DateTimeOffset CreatedAt,
    int HardwareCount,
    int AcceptedShares,
    int RejectedShares);

/// <summary>The mining session currently running on a device, if any.</summary>
public sealed record AdminHardwareSessionResponse(
    Guid SessionId,
    string Status,
    Guid PoolId,
    string PoolName,
    Guid CoinId,
    string CoinCode,
    DateTimeOffset StartedAt,
    DateTimeOffset LastActivityAt);

/// <summary>
/// A mining device owned by a user. <paramref name="HardwareId"/> is the client-generated device
/// id and <paramref name="ConnectedAt"/> is when the device last checked in.
/// </summary>
public sealed record AdminUserHardwareResponse(
    Guid Id,
    Guid HardwareId,
    string? Name,
    string Status,
    DateTimeOffset CreatedAt,
    DateTimeOffset? ConnectedAt,
    int AcceptedShares,
    int RejectedShares,
    AdminHardwareSessionResponse? ActiveSession);

/// <summary>A user account together with every device it owns and what each is mining.</summary>
public sealed record AdminUserDetailResponse(
    Guid Id,
    string Email,
    string? DisplayName,
    string Status,
    DateTimeOffset? EmailConfirmedAt,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt,
    int HardwareCount,
    int AcceptedShares,
    int RejectedShares,
    IReadOnlyList<AdminUserHardwareResponse> Hardware);
