using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Infrastructure.Persistence.Configurations;

internal sealed class CoinConfiguration : IEntityTypeConfiguration<Coin>
{
    public void Configure(EntityTypeBuilder<Coin> builder)
    {
        builder.HasKey(coin => coin.Id);

        builder.Property(coin => coin.Code).IsRequired().HasMaxLength(32);
        builder.HasIndex(coin => coin.Code).IsUnique();

        builder.Property(coin => coin.Name).IsRequired().HasMaxLength(128);
        builder.Property(coin => coin.Network).IsRequired().HasMaxLength(64);

        builder.Property(coin => coin.LastKnownUsdValue).HasPrecision(28, 10);

        builder.Property(coin => coin.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => MiningStatusDbValues.FromDbValue(value));

        builder.Property(coin => coin.CreatedAt).IsRequired();
        builder.Property(coin => coin.UpdatedAt).IsRequired();
    }
}

internal sealed class CoinPriceHistoryConfiguration : IEntityTypeConfiguration<CoinPriceHistory>
{
    public void Configure(EntityTypeBuilder<CoinPriceHistory> builder)
    {
        builder.HasKey(pricePoint => pricePoint.Id);

        builder.Property(pricePoint => pricePoint.UsdValue).HasPrecision(28, 10).IsRequired();
        builder.Property(pricePoint => pricePoint.ObservedAt).IsRequired();
        builder.Property(pricePoint => pricePoint.Source).IsRequired().HasMaxLength(64);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(pricePoint => pricePoint.CoinId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasIndex(pricePoint => new { pricePoint.CoinId, pricePoint.ObservedAt });
    }
}

internal sealed class MiningAlgoConfiguration : IEntityTypeConfiguration<MiningAlgo>
{
    public void Configure(EntityTypeBuilder<MiningAlgo> builder)
    {
        builder.HasKey(algorithm => algorithm.Id);

        builder.Property(algorithm => algorithm.Code).IsRequired().HasMaxLength(64);
        builder.HasIndex(algorithm => algorithm.Code).IsUnique();

        builder.Property(algorithm => algorithm.Name).IsRequired().HasMaxLength(128);

        builder.Property(algorithm => algorithm.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => MiningStatusDbValues.FromDbValue(value));

        builder.Property(algorithm => algorithm.Priority).IsRequired();

        // Algorithm launch templates are free-form JSON, so the shape can evolve per algorithm.
        builder.Property(algorithm => algorithm.Configuration).HasColumnType("jsonb");

        builder.Property(algorithm => algorithm.CreatedAt).IsRequired();
        builder.Property(algorithm => algorithm.UpdatedAt).IsRequired();

        builder.HasIndex(algorithm => algorithm.Priority);
    }
}
