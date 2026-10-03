using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Authentication.Models;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Authentication.Queries;

/// <summary>Newest accounts first, with their device count and terminal share counts.</summary>
public sealed record ListAdminUsersQuery(int Limit) : IRequest<IReadOnlyList<AdminUserSummaryDto>>;

internal sealed class ListAdminUsersQueryHandler(
    IUserRepository users,
    IUserHardwareRepository hardware,
    IShareRepository shares)
    : IRequestHandler<ListAdminUsersQuery, IReadOnlyList<AdminUserSummaryDto>>
{
    public async Task<IReadOnlyList<AdminUserSummaryDto>> Handle(
        ListAdminUsersQuery request,
        CancellationToken cancellationToken)
    {
        var loaded = await users.ListAsync(Math.Clamp(request.Limit, 1, 500), cancellationToken);

        if (loaded.Count == 0)
        {
            return [];
        }

        var userIds = loaded.Select(user => user.Id).ToList();
        var hardwareCounts = await hardware.CountByUsersAsync(userIds, cancellationToken);
        var shareCounts = await shares.CountByUsersAsync(userIds, cancellationToken);

        return loaded
            .Select(user =>
            {
                var counts = shareCounts.GetValueOrDefault(user.Id);

                return user.ToSummaryDto(
                    hardwareCounts.GetValueOrDefault(user.Id),
                    counts.Accepted,
                    counts.Rejected);
            })
            .ToList();
    }
}

/// <summary>A user account together with every device it owns and the session running on it.</summary>
public sealed record GetAdminUserQuery(Guid Id) : IRequest<AdminUserDetailDto>;

internal sealed class GetAdminUserQueryHandler(
    IUserRepository users,
    IUserHardwareRepository hardware,
    IMiningSessionRepository sessions,
    IShareRepository shares,
    ICoinRepository coins,
    IPoolRepository pools)
    : IRequestHandler<GetAdminUserQuery, AdminUserDetailDto>
{
    public async Task<AdminUserDetailDto> Handle(
        GetAdminUserQuery request,
        CancellationToken cancellationToken)
    {
        var user = await users.GetByIdAsync(request.Id, cancellationToken)
            ?? throw new NotFoundException("User not found.");

        var devices = await hardware.ListByUserAsync(user.Id, cancellationToken);
        var deviceIds = devices.Select(device => device.Id).ToList();

        var activeSessions = await sessions.ListActiveByHardwareIdsAsync(deviceIds, cancellationToken);
        var shareCounts = await shares.CountByHardwareIdsAsync(deviceIds, cancellationToken);

        var coinCodes = (await coins.ListAsync(cancellationToken))
            .ToDictionary(coin => coin.Id, coin => coin.Code);
        var poolNames = (await pools.ListAsync(cancellationToken))
            .ToDictionary(pool => pool.Id, pool => pool.Name);

        var sessionsByDevice = activeSessions.ToDictionary(session => session.UserHardwareId);

        var hardwareDtos = devices
            .OrderByDescending(device => device.LastSeenAt ?? device.CreatedAt)
            .Select(device =>
            {
                var counts = shareCounts.GetValueOrDefault(device.Id);

                AdminHardwareSessionDto? session = null;

                if (sessionsByDevice.TryGetValue(device.Id, out var active))
                {
                    session = new AdminHardwareSessionDto(
                        active.Id,
                        active.Status.ToDbValue(),
                        active.PoolId,
                        poolNames.GetValueOrDefault(active.PoolId, "—"),
                        active.CoinId,
                        coinCodes.GetValueOrDefault(active.CoinId, "—"),
                        active.StartedAt,
                        active.LastActivityAt);
                }

                return device.ToDto(counts.Accepted, counts.Rejected, session);
            })
            .ToList();

        return user.ToDetailDto(
            hardwareDtos.Sum(dto => dto.AcceptedShares),
            hardwareDtos.Sum(dto => dto.RejectedShares),
            hardwareDtos);
    }
}
