using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using TokenMiner.Application.Authentication;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Configuration.Abstractions;
using TokenMiner.Application.Mining;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Providers;
using TokenMiner.Application.Providers.Abstractions;
using TokenMiner.Application.Treasury;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Application.Treasury.Commands;
using TokenMiner.Infrastructure.Authentication;
using TokenMiner.Infrastructure.BackgroundJobs;
using TokenMiner.Infrastructure.Integrations;
using TokenMiner.Infrastructure.Persistence;
using TokenMiner.Infrastructure.Persistence.Repositories;

namespace TokenMiner.Infrastructure;

public static class DependencyInjection
{
    /// <summary>
    /// Registers infrastructure services: persistence, authentication primitives and the
    /// email transport.
    /// </summary>
    public static IServiceCollection AddInfrastructure(
        this IServiceCollection services,
        IConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configuration);

        var connectionString = configuration.GetConnectionString("Default");
        if (string.IsNullOrWhiteSpace(connectionString))
        {
            throw new InvalidOperationException(
                "Connection string 'Default' is not configured. Set ConnectionStrings__Default " +
                "(or appsettings.Development.json) before starting the API.");
        }

        var authOptions = configuration.GetSection(AuthOptions.SectionName).Get<AuthOptions>()
            ?? new AuthOptions();

        if (string.IsNullOrWhiteSpace(authOptions.Jwt.SigningKey) || authOptions.Jwt.SigningKey.Length < 32)
        {
            throw new InvalidOperationException(
                "Auth:Jwt:SigningKey must be configured with at least 32 characters.");
        }

        var smtpOptions = configuration.GetSection(SmtpOptions.SectionName).Get<SmtpOptions>()
            ?? new SmtpOptions();

        services.AddSingleton(authOptions);
        services.AddSingleton(smtpOptions);

        var miningOptions = configuration.GetSection(MiningOptions.SectionName).Get<MiningOptions>()
            ?? new MiningOptions();

        services.AddSingleton(miningOptions);

        var treasuryOptions = configuration.GetSection(TreasuryOptions.SectionName).Get<TreasuryOptions>()
            ?? new TreasuryOptions();

        services.AddSingleton(treasuryOptions);

        var providerOptions = configuration.GetSection(ProviderOptions.SectionName).Get<ProviderOptions>()
            ?? new ProviderOptions();

        services.AddSingleton(providerOptions);

        services.AddDbContext<AppDbContext>(options => options
            .UseNpgsql(connectionString, npgsql =>
                npgsql.MigrationsAssembly(typeof(AppDbContext).Assembly.GetName().Name))
            .UseSnakeCaseNamingConvention());

        services.AddScoped<IUserRepository, UserRepository>();
        services.AddScoped<IRefreshTokenStore, RefreshTokenStore>();
        services.AddScoped<IOtpStore, OtpStore>();
        services.AddScoped<IEmailVerificationTokenStore, EmailVerificationTokenStore>();
        services.AddScoped<ILoginAttemptStore, LoginAttemptStore>();
        services.AddScoped<IUnitOfWork, UnitOfWork>();

        services.AddScoped<ICoinRepository, CoinRepository>();
        services.AddScoped<IPoolRepository, PoolRepository>();
        services.AddScoped<IMiningAlgoRepository, MiningAlgoRepository>();

        services.AddScoped<IUserHardwareRepository, UserHardwareRepository>();
        services.AddScoped<IMiningSessionRepository, MiningSessionRepository>();
        services.AddScoped<IMiningLogRepository, MiningLogRepository>();

        services.AddScoped<IShareRepository, ShareRepository>();
        services.AddScoped<IMiningStatisticRepository, MiningStatisticRepository>();
        services.AddScoped<IServiceNonceStore, ServiceNonceStore>();

        services.AddScoped<IPoolPayoutRepository, PoolPayoutRepository>();
        services.AddScoped<IConversionRepository, ConversionRepository>();
        services.AddScoped<ConversionRouteSelector>();

        // Payout integrations. Registering several is enough: the monitor matches on Name.
        services.AddHttpClient<KryptexPoolPayoutProvider>();
        services.AddHttpClient<KryptexCoinPriceProvider>();

        services.AddScoped<IPoolPayoutProvider>(provider =>
            provider.GetRequiredService<KryptexPoolPayoutProvider>());

        services.AddSingleton<IPoolWithdrawalClient>(provider =>
            treasuryOptions.EnableBrowserWithdrawals
                ? ActivatorUtilities.CreateInstance<BrowserPoolWithdrawalClient>(provider)
                : new DisabledPoolWithdrawalClient());

        services.AddSingleton<ICoinPriceProvider>(provider =>
            provider.GetRequiredService<KryptexCoinPriceProvider>());

        // Conversion adapters. The monitor-like pipeline matches them by Name.
        services.AddScoped<IConversionProvider, SimulatedConversionProvider>();
        services.AddScoped<IConversionProvider, ManualConversionProvider>();

        services.AddScoped<ILlmProviderRepository, LlmProviderRepository>();
        services.AddScoped<IProviderDepositRepository, ProviderDepositRepository>();

        services.AddScoped<ISystemConfigurationStore, SystemConfigurationStore>();

        services.AddHttpClient<OpenAiCompatibleLlmProviderClient>();
        services.AddSingleton<ILlmProviderClient>(provider =>
            provider.GetRequiredService<OpenAiCompatibleLlmProviderClient>());

        services.AddSingleton<ISecretResolver, ConfigurationSecretResolver>();
        services.AddSingleton<IBlockchainTransferProvider, SimulatedBlockchainTransferProvider>();

        services.AddSingleton<IPasswordHasher, Pbkdf2PasswordHasher>();
        services.AddSingleton<ISecureTokenService, SecureTokenService>();
        services.AddSingleton<IOtpService, OtpService>();
        services.AddSingleton<IAccessTokenFactory, JwtAccessTokenFactory>();
        services.AddSingleton<IGoogleTokenValidator, GoogleTokenValidator>();
        services.AddScoped<AdminRoleBootstrapper>();

        if (smtpOptions.IsConfigured)
        {
            services.AddScoped<IEmailSender, SmtpEmailSender>();
        }
        else
        {
            services.AddScoped<IEmailSender, NullEmailSender>();
        }

        services.AddHostedService<ShareRewardProcessorJob>();
        services.AddHostedService<MiningStatisticsRollupJob>();
        services.AddHostedService<PoolMonitorJob>();
        services.AddHostedService<PayoutReconciliationJob>();
        services.AddHostedService<CoinPriceJob>();
        services.AddHostedService<ConversionJob>();
        services.AddHostedService<ProviderBalanceJob>();
        services.AddHostedService<ModelCatalogSyncJob>();
        services.AddHostedService<ProviderDepositPipelineJob>();
        services.AddHostedService<AutoTopUpJob>();

        return services;
    }
}
