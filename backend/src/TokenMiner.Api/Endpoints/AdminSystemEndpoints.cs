using MediatR;
using TokenMiner.Application.Configuration.Commands;
using TokenMiner.Contracts.System;

namespace TokenMiner.Api.Endpoints;

/// <summary>
/// Operational surface: a status snapshot for dashboards, and the settings that steer the
/// background services.
/// </summary>
public static class AdminSystemEndpoints
{
    public static IEndpointRouteBuilder MapAdminSystemEndpoints(this IEndpointRouteBuilder app)
    {
        ArgumentNullException.ThrowIfNull(app);

        var group = app.MapGroup("/api/admin/system")
            .WithTags("Admin - System")
            .RequireAuthorization(AuthorizationPolicies.Admin);

        group.MapGet("/status", async (ISender sender, CancellationToken cancellationToken) =>
        {
            var status = await sender.Send(new GetSystemStatusQuery(), cancellationToken);

            return Results.Ok(new SystemStatusResponse(
                status.RunningSessions,
                status.IdleSessions,
                status.PendingShares,
                status.ConversionsInFlight,
                status.ProviderDepositsInFlight,
                status.ProviderBalance,
                status.ProviderReservedBalance,
                status.ProviderBalanceCheckedAt,
                status.CapturedAt));
        })
        .WithName("AdminGetSystemStatus")
        .Produces<SystemStatusResponse>();

        group.MapGet("/configurations", async (ISender sender, CancellationToken cancellationToken) =>
        {
            var settings = await sender.Send(new ListSystemConfigurationsQuery(), cancellationToken);
            return Results.Ok(settings.Select(ToResponse).ToList());
        })
        .WithName("AdminListSystemConfigurations")
        .Produces<IReadOnlyList<SystemConfigurationResponse>>();

        group.MapPut("/configurations/{key}", async (
            string key,
            SetSystemConfigurationRequest request,
            ISender sender,
            CancellationToken cancellationToken) =>
        {
            var setting = await sender.Send(
                new SetSystemConfigurationCommand(key, request.Value),
                cancellationToken);

            return Results.Ok(ToResponse(setting));
        })
        .WithName("AdminSetSystemConfiguration")
        .Produces<SystemConfigurationResponse>();

        return app;
    }

    private static SystemConfigurationResponse ToResponse(SystemConfigurationDto setting) =>
        new(setting.Key, setting.Value, setting.UpdatedAt);
}
