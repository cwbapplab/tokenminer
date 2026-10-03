using MediatR;
using TokenMiner.Application.Authentication.Models;
using TokenMiner.Application.Authentication.Queries;
using TokenMiner.Contracts.Admin;

namespace TokenMiner.Api.Endpoints;

/// <summary>
/// Read-only operator views over user accounts, the mining devices each owns and the shares those
/// devices have produced. All routes require the <c>admin</c> role.
/// </summary>
public static class AdminUserEndpoints
{
    public static IEndpointRouteBuilder MapAdminUserEndpoints(this IEndpointRouteBuilder app)
    {
        ArgumentNullException.ThrowIfNull(app);

        var users = app.MapGroup("/api/admin/users")
            .WithTags("Admin - Users")
            .RequireAuthorization(AuthorizationPolicies.Admin);

        users.MapGet(string.Empty, ListUsersAsync)
            .WithName("AdminListUsers")
            .Produces<IReadOnlyList<AdminUserSummaryResponse>>();

        users.MapGet("/{id:guid}", GetUserAsync)
            .WithName("AdminGetUser")
            .Produces<AdminUserDetailResponse>()
            .ProducesProblem(StatusCodes.Status404NotFound);

        return app;
    }

    private static async Task<IResult> ListUsersAsync(
        int? limit,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var users = await sender.Send(new ListAdminUsersQuery(limit ?? 100), cancellationToken);

        return Results.Ok(users.Select(ToResponse).ToList());
    }

    private static async Task<IResult> GetUserAsync(
        Guid id,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var user = await sender.Send(new GetAdminUserQuery(id), cancellationToken);

        return Results.Ok(ToResponse(user));
    }

    private static AdminUserSummaryResponse ToResponse(AdminUserSummaryDto user) => new(
        user.Id,
        user.Email,
        user.DisplayName,
        user.Status,
        user.CreatedAt,
        user.HardwareCount,
        user.AcceptedShares,
        user.RejectedShares);

    private static AdminUserDetailResponse ToResponse(AdminUserDetailDto user) => new(
        user.Id,
        user.Email,
        user.DisplayName,
        user.Status,
        user.EmailConfirmedAt,
        user.CreatedAt,
        user.UpdatedAt,
        user.HardwareCount,
        user.AcceptedShares,
        user.RejectedShares,
        user.Hardware.Select(ToResponse).ToList());

    private static AdminUserHardwareResponse ToResponse(AdminUserHardwareDto hardware) => new(
        hardware.Id,
        hardware.HardwareId,
        hardware.Name,
        hardware.Status,
        hardware.CreatedAt,
        hardware.ConnectedAt,
        hardware.AcceptedShares,
        hardware.RejectedShares,
        hardware.ActiveSession is null ? null : ToResponse(hardware.ActiveSession));

    private static AdminHardwareSessionResponse ToResponse(AdminHardwareSessionDto session) => new(
        session.SessionId,
        session.Status,
        session.PoolId,
        session.PoolName,
        session.CoinId,
        session.CoinCode,
        session.StartedAt,
        session.LastActivityAt);
}
