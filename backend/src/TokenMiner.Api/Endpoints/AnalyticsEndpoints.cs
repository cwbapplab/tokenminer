using System.Security.Claims;
using MediatR;
using TokenMiner.Application.Mining.Queries;
using TokenMiner.Contracts.Mining;

namespace TokenMiner.Api.Endpoints;

public static class AnalyticsEndpoints
{
    public static IEndpointRouteBuilder MapAnalyticsEndpoints(this IEndpointRouteBuilder app)
    {
        ArgumentNullException.ThrowIfNull(app);

        app.MapGet("/api/analytics", GetAnalyticsAsync)
            .WithTags("Mining")
            .WithName("GetAnalytics")
            .RequireAuthorization(AuthorizationPolicies.User)
            .Produces<MiningAnalyticsResponse>()
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        return app;
    }

    private static async Task<IResult> GetAnalyticsAsync(
        Guid? userHardwareId,
        ClaimsPrincipal user,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var userId = user.GetUserId();
        if (userId is null)
        {
            return Results.Unauthorized();
        }

        // The device filter is applied inside the caller's own rows, never as a lookup.
        var analytics = await sender.Send(new GetAnalyticsQuery(userId.Value, userHardwareId), cancellationToken);

        return Results.Ok(new MiningAnalyticsResponse(
            analytics.Hardware
                .Select(item => new HardwareAnalyticsResponse(
                    item.UserHardwareId,
                    item.Last30Usd,
                    item.Last60Usd,
                    item.AllTimeUsd))
                .ToList(),
            new AnalyticsTotalsResponse(
                analytics.Totals.Last30Usd,
                analytics.Totals.Last60Usd,
                analytics.Totals.AllTimeUsd)));
    }
}
