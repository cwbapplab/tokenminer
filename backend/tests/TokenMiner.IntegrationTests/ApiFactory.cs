using System.Net.Http.Headers;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Testcontainers.PostgreSql;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Providers.Abstractions;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Contracts.Auth;
using TokenMiner.Domain.Users;
using TokenMiner.Infrastructure.Persistence;
using Xunit;

namespace TokenMiner.IntegrationTests;

/// <summary>
/// Boots the real API against a disposable PostgreSQL container and swaps the email
/// transport for a capturing double.
/// </summary>
/// <remarks>
/// Settings are supplied through environment variables rather than
/// <c>ConfigureAppConfiguration</c>: with minimal hosting that callback is applied during
/// <c>Build()</c>, which is after <c>Program.cs</c> has already read configuration. Environment
/// variables are picked up when the builder is created, so they are visible in time.
/// </remarks>
public sealed class ApiFactory : WebApplicationFactory<Program>, IAsyncLifetime
{
    private readonly PostgreSqlContainer _postgres = new PostgreSqlBuilder("postgres:16")
        .WithDatabase("tokenminer_tests")
        .WithUsername("tokenminer")
        .WithPassword("tokenminer")
        .Build();

    private readonly Dictionary<string, string?> _previousEnvironment = [];

    public CapturingEmailSender Email { get; } = new();

    /// <summary>Stand-in pool integration; tests script what the pool reports.</summary>
    public StubPoolPayoutProvider PoolPayouts { get; } = new();

    public StubCoinPriceProvider CoinPrices { get; } = new();

    /// <summary>Stand-in LLM gateway; tests script its balance and catalogue.</summary>
    public StubLlmProviderClient LlmProvider { get; } = new();

    async Task IAsyncLifetime.InitializeAsync()
    {
        await _postgres.StartAsync();

        ApplyEnvironment(new Dictionary<string, string?>
        {
            ["ConnectionStrings__Default"] = _postgres.GetConnectionString(),
            ["Auth__Jwt__SigningKey"] = "integration-test-signing-key-at-least-32-characters",
            ["Auth__Jwt__Issuer"] = "tokenminer-integration-tests",
            ["Auth__Jwt__Audience"] = "tokenminer-integration-tests",
            ["Auth__Jwt__KeyId"] = "test",
            ["Auth__Jwt__AccessTokenMinutes"] = "15",
            ["Auth__Otp__Length"] = "6",
            ["Auth__Otp__ExpiryMinutes"] = "10",
            ["Auth__Otp__MaxAttempts"] = "5",
            ["Auth__Otp__ResendCooldownSeconds"] = "60",
            ["Auth__Otp__HmacKey"] = "integration-test-otp-hmac-key",
            ["Auth__Lockout__MaxFailedAttempts"] = "3",
            ["Auth__Lockout__WindowMinutes"] = "15",
            // High enough that the per-IP throttle never interferes with the suite.
            ["Auth__RateLimit__PermitLimit"] = "10000",
            ["Auth__RateLimit__WindowSeconds"] = "60",
            ["Auth__RequireOtpForOauth"] = "true",
            ["Smtp__Host"] = string.Empty,
            // Push the periodic jobs far into the future so tests drive each step explicitly.
            ["Mining__WatchdogIntervalSeconds"] = "3600",
            ["Mining__HeartbeatGraceSeconds"] = "10",
            // The public proxy endpoint clients must connect to; sessions refuse to start without it.
            ["Mining__PublicStratumEndpoint"] = "localhost:3333",
            ["Mining__ShareProcessorIntervalSeconds"] = "3600",
            ["Mining__StatisticsIntervalSeconds"] = "3600",
            ["Mining__ServiceSharedSecret"] = MiningTestData.ServiceSharedSecret,
            // Treasury jobs are driven explicitly by the treasury tests.
            ["Treasury__PoolMonitorIntervalSeconds"] = "3600",
            ["Treasury__PayoutReconciliationIntervalSeconds"] = "3600",
            ["Treasury__CoinPriceIntervalSeconds"] = "3600",
            // Provider jobs are driven explicitly too; the simulated chain settles immediately.
            ["Providers__BalanceIntervalSeconds"] = "3600",
            ["Providers__ModelCatalogIntervalSeconds"] = "3600",
            ["Providers__DepositIntervalSeconds"] = "3600",
            ["Providers__ConfirmationThreshold"] = "6",
            ["Providers__SimulatedConfirmations"] = "10",
        });

        using var scope = Services.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        await dbContext.Database.MigrateAsync();
    }

    async Task IAsyncLifetime.DisposeAsync()
    {
        await base.DisposeAsync();
        await _postgres.DisposeAsync();

        foreach (var (key, value) in _previousEnvironment)
        {
            Environment.SetEnvironmentVariable(key, value);
        }
    }

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Testing");

        builder.ConfigureServices(services =>
        {
            services.RemoveAll<IEmailSender>();
            services.AddSingleton<IEmailSender>(Email);

            services.RemoveAll<IPoolPayoutProvider>();
            services.AddSingleton<IPoolPayoutProvider>(PoolPayouts);

            services.RemoveAll<ICoinPriceProvider>();
            services.AddSingleton<ICoinPriceProvider>(CoinPrices);

            services.RemoveAll<ILlmProviderClient>();
            services.AddSingleton<ILlmProviderClient>(LlmProvider);
        });
    }

    /// <summary>Registers and activates a fresh account.</summary>
    public async Task<(HttpClient Client, string Email, string Password)> CreateActivatedUserAsync()
    {
        const string password = "Str0ngPassword!23";

        var client = CreateClient();
        var email = $"user-{Guid.NewGuid():N}@tokenminer.local";

        var register = await client.PostAsJsonAsync(
            "/api/auth/register",
            new RegisterRequest(email, password, "Test User"));
        register.EnsureSuccessStatusCode();

        var activate = await client.PostAsJsonAsync(
            "/api/auth/otp/verify",
            new VerifyOtpRequest(email, Email.LatestActivationCode(email)));
        activate.EnsureSuccessStatusCode();

        return (client, email, password);
    }

    /// <summary>An authenticated client for an ordinary (non-admin) account.</summary>
    public async Task<HttpClient> CreateUserClientAsync()
    {
        var (client, email, password) = await CreateActivatedUserAsync();
        await AuthenticateAsync(client, email, password);
        return client;
    }

    /// <summary>An authenticated client together with its access token, for WebSocket tests.</summary>
    public async Task<(HttpClient Client, string AccessToken)> CreateUserClientWithTokenAsync()
    {
        var (client, email, password) = await CreateActivatedUserAsync();
        var accessToken = await AuthenticateAsync(client, email, password);
        return (client, accessToken);
    }

    /// <summary>
    /// An authenticated client whose account holds the admin role. The role is granted
    /// directly in the database before signing in, so the issued token carries it.
    /// </summary>
    public async Task<HttpClient> CreateAdminClientAsync()
    {
        var (client, email, password) = await CreateActivatedUserAsync();

        await using (var scope = Services.CreateAsyncScope())
        {
            var dbContext = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            var user = await dbContext.Users.SingleAsync(candidate => candidate.NormalizedEmail == email.ToUpperInvariant());

            dbContext.UserRoles.Add(new UserRole(user.Id, RoleIds.Admin));
            await dbContext.SaveChangesAsync();
        }

        await AuthenticateAsync(client, email, password);
        return client;
    }

    private static async Task<string> AuthenticateAsync(HttpClient client, string email, string password)
    {
        var login = await client.PostAsJsonAsync("/api/auth/login", new LoginRequest(email, password));
        login.EnsureSuccessStatusCode();

        var tokens = await login.Content.ReadFromJsonAsync<AuthTokensResponse>();
        client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", tokens!.AccessToken);

        return tokens.AccessToken;
    }

    private void ApplyEnvironment(IReadOnlyDictionary<string, string?> values)
    {
        foreach (var (key, value) in values)
        {
            _previousEnvironment[key] = Environment.GetEnvironmentVariable(key);
            Environment.SetEnvironmentVariable(key, value);
        }
    }
}

[CollectionDefinition(ApiCollection.Name)]
public sealed class ApiCollection : ICollectionFixture<ApiFactory>
{
    public const string Name = "api";
}
