using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Domain.Users;

namespace TokenMiner.Infrastructure.Persistence.Configurations;

internal sealed class UserMiningShareConfiguration : IEntityTypeConfiguration<UserMiningShare>
{
    public void Configure(EntityTypeBuilder<UserMiningShare> builder)
    {
        builder.HasKey(share => share.Id);

        builder.Property(share => share.ShareIdentifier).IsRequired().HasMaxLength(256);
        builder.Property(share => share.WorkerIdentifier).IsRequired().HasMaxLength(128);

        builder.Property(share => share.JobId).HasMaxLength(128);
        builder.Property(share => share.Nonce).HasMaxLength(128);
        builder.Property(share => share.Extranonce).HasMaxLength(128);
        builder.Property(share => share.Target).HasMaxLength(128);
        builder.Property(share => share.ResultHash).HasMaxLength(256);

        builder.Property(share => share.Difficulty).HasPrecision(28, 10);
        builder.Property(share => share.CoinValue).HasPrecision(28, 10);
        builder.Property(share => share.ApproxUsdValueAtTime).HasPrecision(28, 10);

        // The pool's own response is kept verbatim so a share can be re-validated later.
        builder.Property(share => share.PoolResponse).HasColumnType("jsonb");

        builder.Property(share => share.RewardStatus)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => ShareRewardStatusDbValues.FromDbValue(value));

        builder.Property(share => share.Reason).HasMaxLength(512);

        builder.Property(share => share.ShareTimestamp).IsRequired();
        builder.Property(share => share.CreatedAt).IsRequired();

        builder.HasOne<User>()
            .WithMany()
            .HasForeignKey(share => share.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<UserHardware>()
            .WithMany()
            .HasForeignKey(share => share.UserHardwareId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<UserHardwareMiner>()
            .WithMany()
            .HasForeignKey(share => share.UserHardwareMinerId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<Pool>()
            .WithMany()
            .HasForeignKey(share => share.PoolId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(share => share.CoinId)
            .OnDelete(DeleteBehavior.Restrict);

        // Idempotency: a pool-side share can only ever be stored once.
        builder.HasIndex(share => new { share.PoolId, share.ShareIdentifier }).IsUnique();

        builder.HasIndex(share => new { share.UserId, share.CreatedAt });
        builder.HasIndex(share => new { share.RewardStatus, share.CreatedAt });
        builder.HasIndex(share => share.UserHardwareId);
    }
}

internal sealed class MiningStatisticConfiguration : IEntityTypeConfiguration<MiningStatistic>
{
    public void Configure(EntityTypeBuilder<MiningStatistic> builder)
    {
        builder.HasKey(statistic => statistic.Id);

        builder.Property(statistic => statistic.Period)
            .IsRequired()
            .HasMaxLength(16)
            .HasConversion(
                period => period.ToDbValue(),
                value => StatisticPeriodDbValues.FromDbValue(value));

        builder.Property(statistic => statistic.Amount).HasPrecision(28, 10).IsRequired();
        builder.Property(statistic => statistic.UsdValue).HasPrecision(28, 10).IsRequired();
        builder.Property(statistic => statistic.CalculatedAt).IsRequired();

        builder.HasOne<User>()
            .WithMany()
            .HasForeignKey(statistic => statistic.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<UserHardware>()
            .WithMany()
            .HasForeignKey(statistic => statistic.UserHardwareId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(statistic => statistic.CoinId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasIndex(statistic => new { statistic.UserHardwareId, statistic.CoinId, statistic.Period })
            .IsUnique();

        builder.HasIndex(statistic => statistic.UserId);
    }
}

internal sealed class ServiceRequestNonceConfiguration : IEntityTypeConfiguration<ServiceRequestNonce>
{
    public void Configure(EntityTypeBuilder<ServiceRequestNonce> builder)
    {
        builder.HasKey(nonce => new { nonce.ServiceId, nonce.Nonce });

        builder.Property(nonce => nonce.ServiceId).HasMaxLength(64);
        builder.Property(nonce => nonce.Nonce).HasMaxLength(128);

        builder.Property(nonce => nonce.ExpiresAt).IsRequired();
        builder.Property(nonce => nonce.CreatedAt).IsRequired();

        builder.HasIndex(nonce => nonce.ExpiresAt);
    }
}
