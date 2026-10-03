using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Domain.Users;
using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Application.Authentication.Models;

/// <summary>
/// Read model for the operator's user list. The share counts are the terminal reward states:
/// accepted (including redeemed) and rejected.
/// </summary>
public sealed record AdminUserSummaryDto(
    Guid Id,
    string Email,
    string? DisplayName,
    string Status,
    DateTimeOffset CreatedAt,
    int HardwareCount,
    int AcceptedShares,
    int RejectedShares);

public sealed record AdminHardwareSessionDto(
    Guid SessionId,
    string Status,
    Guid PoolId,
    string PoolName,
    Guid CoinId,
    string CoinCode,
    DateTimeOffset StartedAt,
    DateTimeOffset LastActivityAt);

public sealed record AdminUserHardwareDto(
    Guid Id,
    Guid HardwareId,
    string? Name,
    string Status,
    DateTimeOffset CreatedAt,
    DateTimeOffset? ConnectedAt,
    int AcceptedShares,
    int RejectedShares,
    AdminHardwareSessionDto? ActiveSession);

public sealed record AdminUserDetailDto(
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
    IReadOnlyList<AdminUserHardwareDto> Hardware);

public static class AdminUserMappings
{
    public static AdminUserSummaryDto ToSummaryDto(
        this User user,
        int hardwareCount,
        int acceptedShares,
        int rejectedShares) => new(
        user.Id,
        user.Email,
        user.DisplayName,
        user.Status.ToDbValue(),
        user.CreatedAt,
        hardwareCount,
        acceptedShares,
        rejectedShares);

    public static AdminUserDetailDto ToDetailDto(
        this User user,
        int acceptedShares,
        int rejectedShares,
        IReadOnlyList<AdminUserHardwareDto> hardware) => new(
        user.Id,
        user.Email,
        user.DisplayName,
        user.Status.ToDbValue(),
        user.EmailConfirmedAt,
        user.CreatedAt,
        user.UpdatedAt,
        hardware.Count,
        acceptedShares,
        rejectedShares,
        hardware);

    public static AdminUserHardwareDto ToDto(
        this UserHardware hardware,
        int acceptedShares,
        int rejectedShares,
        AdminHardwareSessionDto? activeSession) => new(
        hardware.Id,
        hardware.HardwareId,
        hardware.Name,
        hardware.Status.ToDbValue(),
        hardware.CreatedAt,
        hardware.LastSeenAt,
        acceptedShares,
        rejectedShares,
        activeSession);
}
