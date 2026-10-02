using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;

namespace TokenMiner.Infrastructure.Persistence.Configurations;

internal sealed class PoolPayoutConfiguration : IEntityTypeConfiguration<PoolPayout>
{
    public void Configure(EntityTypeBuilder<PoolPayout> builder)
    {
        builder.HasKey(payout => payout.Id);

        builder.Property(payout => payout.WalletAddress).HasMaxLength(256);
        builder.Property(payout => payout.Amount).HasPrecision(28, 10).IsRequired();
        builder.Property(payout => payout.RedeemedAmount).HasPrecision(28, 10).IsRequired();
        builder.Property(payout => payout.TransactionHash).HasMaxLength(256);

        builder.Property(payout => payout.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => PoolPayoutStatusDbValues.FromDbValue(value));

        // The pool's own record of the payout, kept verbatim.
        builder.Property(payout => payout.RawData).HasColumnType("jsonb");

        builder.Property(payout => payout.CreatedAt).IsRequired();
        builder.Property(payout => payout.UpdatedAt).IsRequired();

        builder.HasOne<Pool>()
            .WithMany()
            .HasForeignKey(payout => payout.PoolId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(payout => payout.CoinId)
            .OnDelete(DeleteBehavior.Restrict);

        // Idempotency: a blockchain payout can only be recorded once per pool.
        builder.HasIndex(payout => new { payout.PoolId, payout.TransactionHash }).IsUnique();

        builder.HasIndex(payout => new { payout.Status, payout.UpdatedAt });
    }
}
