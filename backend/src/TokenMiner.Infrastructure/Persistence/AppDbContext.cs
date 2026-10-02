using Microsoft.EntityFrameworkCore;
using TokenMiner.Domain.Configuration;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Providers;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Users;

namespace TokenMiner.Infrastructure.Persistence;

/// <summary>
/// Application database context.
/// Entity configurations live in <c>IEntityTypeConfiguration&lt;T&gt;</c> classes
/// discovered from this assembly; table/column names are snake_cased via
/// <c>UseSnakeCaseNamingConvention()</c> on the options builder.
/// </summary>
public sealed class AppDbContext(DbContextOptions<AppDbContext> options) : DbContext(options)
{
    public DbSet<User> Users => Set<User>();

    public DbSet<Role> Roles => Set<Role>();

    public DbSet<UserRole> UserRoles => Set<UserRole>();

    public DbSet<RefreshToken> RefreshTokens => Set<RefreshToken>();

    public DbSet<OtpCode> OtpCodes => Set<OtpCode>();

    public DbSet<EmailVerificationToken> EmailVerificationTokens => Set<EmailVerificationToken>();

    public DbSet<LoginAttempt> LoginAttempts => Set<LoginAttempt>();

    public DbSet<Coin> Coins => Set<Coin>();

    public DbSet<CoinPriceHistory> CoinPriceHistory => Set<CoinPriceHistory>();

    public DbSet<Pool> Pools => Set<Pool>();

    public DbSet<PoolDetails> PoolDetails => Set<PoolDetails>();

    public DbSet<PoolCoin> PoolCoins => Set<PoolCoin>();

    public DbSet<MiningAlgo> MiningAlgos => Set<MiningAlgo>();

    public DbSet<UserHardware> UserHardware => Set<UserHardware>();

    public DbSet<UserHardwareMiner> UserHardwareMiners => Set<UserHardwareMiner>();

    public DbSet<MiningLog> MiningLogs => Set<MiningLog>();

    public DbSet<UserMiningShare> UserMiningShares => Set<UserMiningShare>();

    public DbSet<MiningStatistic> MiningStatistics => Set<MiningStatistic>();

    public DbSet<PoolPayout> PoolPayouts => Set<PoolPayout>();

    public DbSet<ConversionProvider> ConversionProviders => Set<ConversionProvider>();

    public DbSet<ConversionRoute> ConversionRoutes => Set<ConversionRoute>();

    public DbSet<ConversionTransaction> ConversionTransactions => Set<ConversionTransaction>();

    public DbSet<LlmProvider> LlmProviders => Set<LlmProvider>();

    public DbSet<LlmProviderDepositAccount> LlmProviderDepositAccounts => Set<LlmProviderDepositAccount>();

    public DbSet<LlmProviderModel> LlmProviderModels => Set<LlmProviderModel>();

    public DbSet<ProviderBalanceSnapshot> ProviderBalanceSnapshots => Set<ProviderBalanceSnapshot>();

    public DbSet<ProviderDeposit> ProviderDeposits => Set<ProviderDeposit>();

    public DbSet<SystemConfiguration> SystemConfigurations => Set<SystemConfiguration>();

    internal DbSet<ServiceRequestNonce> ServiceRequestNonces => Set<ServiceRequestNonce>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        ArgumentNullException.ThrowIfNull(modelBuilder);

        modelBuilder.ApplyConfigurationsFromAssembly(typeof(AppDbContext).Assembly);

        base.OnModelCreating(modelBuilder);
    }
}
