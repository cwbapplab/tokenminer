using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Domain.Providers;
using TokenMiner.Domain.Providers.Enums;

namespace TokenMiner.Infrastructure.Persistence.Configurations;

internal sealed class LlmProviderConfiguration : IEntityTypeConfiguration<LlmProvider>
{
    public void Configure(EntityTypeBuilder<LlmProvider> builder)
    {
        builder.HasKey(provider => provider.Id);

        builder.Property(provider => provider.Name).IsRequired().HasMaxLength(64);
        builder.HasIndex(provider => provider.Name).IsUnique();

        builder.Property(provider => provider.Endpoint).IsRequired().HasMaxLength(512);

        // Only a reference into the secret resolver, never the API key itself.
        builder.Property(provider => provider.CredentialRef).HasMaxLength(256);

        builder.Property(provider => provider.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => MiningStatusDbValues.FromDbValue(value));

        builder.Property(provider => provider.CreatedAt).IsRequired();
        builder.Property(provider => provider.UpdatedAt).IsRequired();
    }
}

internal sealed class LlmProviderDepositAccountConfiguration : IEntityTypeConfiguration<LlmProviderDepositAccount>
{
    public void Configure(EntityTypeBuilder<LlmProviderDepositAccount> builder)
    {
        builder.HasKey(account => account.Id);

        builder.Property(account => account.Network).IsRequired().HasMaxLength(64);
        builder.Property(account => account.DepositAddress).IsRequired().HasMaxLength(256);

        builder.Property(account => account.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => MiningStatusDbValues.FromDbValue(value));

        builder.Property(account => account.CreatedAt).IsRequired();
        builder.Property(account => account.UpdatedAt).IsRequired();

        builder.HasOne<LlmProvider>()
            .WithMany()
            .HasForeignKey(account => account.LlmProviderId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(account => account.CoinId)
            .OnDelete(DeleteBehavior.Restrict);

        // A USDT address on BSC is a different rail from the same token on Tron.
        builder.HasIndex(account => new { account.LlmProviderId, account.CoinId, account.Network })
            .IsUnique();
    }
}

internal sealed class LlmProviderModelConfiguration : IEntityTypeConfiguration<LlmProviderModel>
{
    public void Configure(EntityTypeBuilder<LlmProviderModel> builder)
    {
        builder.HasKey(model => model.Id);

        builder.Property(model => model.ModelId).IsRequired().HasMaxLength(200);
        builder.Property(model => model.Name).HasMaxLength(200);

        builder.Property(model => model.InputCost).HasPrecision(28, 10);
        builder.Property(model => model.OutputCost).HasPrecision(28, 10);
        builder.Property(model => model.CachedInputCost).HasPrecision(28, 10);

        builder.Property(model => model.Currency).IsRequired().HasMaxLength(8);

        builder.Property(model => model.Capabilities).HasColumnType("jsonb");

        builder.Property(model => model.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => MiningStatusDbValues.FromDbValue(value));

        builder.Property(model => model.LastSyncedAt).IsRequired();
        builder.Property(model => model.CreatedAt).IsRequired();
        builder.Property(model => model.UpdatedAt).IsRequired();

        builder.HasOne<LlmProvider>()
            .WithMany()
            .HasForeignKey(model => model.LlmProviderId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasIndex(model => new { model.LlmProviderId, model.ModelId }).IsUnique();
    }
}

internal sealed class ProviderBalanceSnapshotConfiguration : IEntityTypeConfiguration<ProviderBalanceSnapshot>
{
    public void Configure(EntityTypeBuilder<ProviderBalanceSnapshot> builder)
    {
        builder.HasKey(snapshot => snapshot.Id);

        builder.Property(snapshot => snapshot.Balance).HasPrecision(28, 10).IsRequired();
        builder.Property(snapshot => snapshot.ReservedBalance).HasPrecision(28, 10).IsRequired();
        builder.Property(snapshot => snapshot.TotalBalance).HasPrecision(28, 10).IsRequired();
        builder.Property(snapshot => snapshot.CheckedAt).IsRequired();

        builder.HasOne<LlmProvider>()
            .WithMany()
            .HasForeignKey(snapshot => snapshot.LlmProviderId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasIndex(snapshot => new { snapshot.LlmProviderId, snapshot.CheckedAt });
    }
}

internal sealed class ProviderDepositConfiguration : IEntityTypeConfiguration<ProviderDeposit>
{
    public void Configure(EntityTypeBuilder<ProviderDeposit> builder)
    {
        builder.HasKey(deposit => deposit.Id);

        builder.Property(deposit => deposit.Network).IsRequired().HasMaxLength(64);
        builder.Property(deposit => deposit.Address).IsRequired().HasMaxLength(256);
        builder.Property(deposit => deposit.Amount).HasPrecision(28, 10).IsRequired();
        builder.Property(deposit => deposit.TransactionHash).HasMaxLength(256);

        builder.Property(deposit => deposit.ProviderCreditBefore).HasPrecision(28, 10);
        builder.Property(deposit => deposit.ProviderCreditAfter).HasPrecision(28, 10);

        builder.Property(deposit => deposit.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => ProviderDepositStatusDbValues.FromDbValue(value));

        // Unique per transfer intent, so a retry cannot send the same funds twice.
        builder.Property(deposit => deposit.IdempotencyKey).IsRequired().HasMaxLength(256);
        builder.HasIndex(deposit => deposit.IdempotencyKey).IsUnique();

        builder.Property(deposit => deposit.Error).HasMaxLength(1024);

        builder.Property(deposit => deposit.CreatedAt).IsRequired();
        builder.Property(deposit => deposit.UpdatedAt).IsRequired();

        builder.HasOne<LlmProvider>()
            .WithMany()
            .HasForeignKey(deposit => deposit.LlmProviderId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(deposit => deposit.CoinId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasIndex(deposit => new { deposit.Status, deposit.UpdatedAt });
    }
}
