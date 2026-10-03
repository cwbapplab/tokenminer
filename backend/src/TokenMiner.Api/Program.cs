using Microsoft.AspNetCore.Diagnostics.HealthChecks;
using Serilog;
using TokenMiner.Api;
using TokenMiner.Api.Endpoints;
using TokenMiner.Application;
using TokenMiner.Application.Authentication;
using TokenMiner.Infrastructure;
using TokenMiner.Infrastructure.Authentication;

var builder = WebApplication.CreateBuilder(args);

builder.Host.UseSerilog((context, services, loggerConfiguration) => loggerConfiguration
    .ReadFrom.Configuration(context.Configuration)
    .ReadFrom.Services(services)
    .Enrich.FromLogContext());

builder.Services.AddApplication();
builder.Services.AddInfrastructure(builder.Configuration);
builder.Services.AddAuth(builder.Configuration);

builder.Services.AddExceptionHandler<AppExceptionHandler>();
builder.Services.AddProblemDetails();

var healthChecks = builder.Services.AddHealthChecks();

var connectionString = builder.Configuration.GetConnectionString("Default");
if (!string.IsNullOrWhiteSpace(connectionString))
{
    healthChecks.AddNpgSql(connectionString, name: "postgres", tags: ["ready"]);
}

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

app.UseSerilogRequestLogging();
app.UseExceptionHandler();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseRateLimiter();

// Machine-to-machine routes authenticate with a signed request. This must run before endpoint
// execution so the request body is still intact for signature verification.
app.UseWhen(
    context => context.Request.Path.StartsWithSegments("/internal"),
    branch => branch.UseMiddleware<ServiceAuthMiddleware>());

app.UseWebSockets();
app.UseAuthentication();
app.UseAuthorization();

app.MapGet("/", () => Results.Ok(new { name = "TokenMiner.Api", status = "ok" }))
    .AllowAnonymous();

app.MapHealthChecks("/health/live", new HealthCheckOptions
{
    Predicate = _ => false,
}).AllowAnonymous();

app.MapHealthChecks("/health/ready", new HealthCheckOptions
{
    Predicate = registration => registration.Tags.Contains("ready"),
}).AllowAnonymous();

app.MapAuthEndpoints();
app.MapAdminMiningEndpoints();
app.MapAdminUserEndpoints();
app.MapAdminTreasuryEndpoints();
app.MapAdminProviderEndpoints();
app.MapAdminSystemEndpoints();
app.MapMiningEndpoints();
app.MapAnalyticsEndpoints();
app.MapInternalMiningEndpoints();
app.MapInternalStratumEndpoints();

// Only touches the database when bootstrap admins are actually configured.
if (app.Services.GetRequiredService<AuthOptions>().BootstrapAdminEmails.Count > 0)
{
    using var scope = app.Services.CreateScope();
    await scope.ServiceProvider.GetRequiredService<AdminRoleBootstrapper>().EnsureAsync(CancellationToken.None);
}

app.Run();

/// <summary>Exposed so the integration test host can reference the API assembly.</summary>
public partial class Program;
