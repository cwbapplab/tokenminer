using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;

namespace TokenMiner.Infrastructure.Persistence.Configurations;

internal sealed class ConversionProviderConfiguration : IEntityTypeConfiguration<ConversionProvider>
{
    public void Configure(EntityTypeBuilder<ConversionProvider> builder)
    {
        builder.HasKey(provider => provider.Id);

        builder.Property(provider => provider.Name).IsRequired().HasMaxLength(64);
        builder.HasIndex(provider => provider.Name).IsUnique();

        builder.Property(provider => provider.BaseUrl).IsRequired().HasMaxLength(512);

        builder.Property(provider => provider.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => TreasuryStatusDbValues.FromDbValue(value));

        // Only a reference into the secret store, never the credential itself.
        builder.Property(provider => provider.CredentialRef).HasMaxLength(256);

        builder.Property(provider => provider.SupportedFeatures).HasColumnType("jsonb");

        builder.Property(provider => provider.CreatedAt).IsRequired();
        builder.Property(provider => provider.UpdatedAt).IsRequired();
    }
}

internal sealed class ConversionRouteConfiguration : IEntityTypeConfiguration<ConversionRoute>
{
    public void Configure(EntityTypeBuilder<ConversionRoute> builder)
    {
        builder.HasKey(route => route.Id);

        builder.Property(route => route.SourceNetwork).HasMaxLength(64);
        builder.Property(route => route.DestinationNetwork).HasMaxLength(64);
        builder.Property(route => route.MinimumAmount).HasPrecision(28, 10).IsRequired();

        builder.Property(route => route.CreatedAt).IsRequired();
        builder.Property(route => route.UpdatedAt).IsRequired();

        builder.HasOne<ConversionProvider>()
            .WithMany()
            .HasForeignKey(route => route.ConversionProviderId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(route => route.SourceCoinId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(route => route.DestinationCoinId)
            .OnDelete(DeleteBehavior.Restrict);

        // One route per provider and pair.
        builder.HasIndex(route => new { route.ConversionProviderId, route.SourceCoinId, route.DestinationCoinId })
            .IsUnique();

        builder.HasIndex(route => new { route.SourceCoinId, route.Enabled, route.Priority });
    }
}

internal sealed class ConversionTransactionConfiguration : IEntityTypeConfiguration<ConversionTransaction>
{
    public void Configure(EntityTypeBuilder<ConversionTransaction> builder)
    {
        builder.HasKey(transaction => transaction.Id);

        builder.Property(transaction => transaction.SourceAmount).HasPrecision(28, 10).IsRequired();
        builder.Property(transaction => transaction.DestinationAmount).HasPrecision(28, 10);
        builder.Property(transaction => transaction.ExchangeRate).HasPrecision(28, 10);
        builder.Property(transaction => transaction.Fees).HasPrecision(28, 10);

        builder.Property(transaction => transaction.SourceTransactionId).HasMaxLength(256);
        builder.Property(transaction => transaction.DestinationTransactionId).HasMaxLength(256);

        builder.Property(transaction => transaction.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => ConversionTransactionStatusDbValues.FromDbValue(value));

        builder.Property(transaction => transaction.IdempotencyKey).IsRequired().HasMaxLength(256);
        builder.HasIndex(transaction => transaction.IdempotencyKey).IsUnique();

        builder.Property(transaction => transaction.Error).HasMaxLength(1024);

        builder.Property(transaction => transaction.CreatedAt).IsRequired();
        builder.Property(transaction => transaction.UpdatedAt).IsRequired();

        builder.HasOne<ConversionProvider>()
            .WithMany()
            .HasForeignKey(transaction => transaction.ConversionProviderId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(transaction => transaction.SourceCoinId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(transaction => transaction.DestinationCoinId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasIndex(transaction => new { transaction.Status, transaction.UpdatedAt });
    }
}
