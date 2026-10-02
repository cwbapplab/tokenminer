using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Domain.Users;

namespace TokenMiner.Infrastructure.Persistence.Configurations;

internal sealed class UserHardwareMinerConfiguration : IEntityTypeConfiguration<UserHardwareMiner>
{
    public void Configure(EntityTypeBuilder<UserHardwareMiner> builder)
    {
        builder.HasKey(session => session.Id);

        builder.Property(session => session.WorkerIdentifier).IsRequired().HasMaxLength(128);
        builder.HasIndex(session => session.WorkerIdentifier);

        builder.Property(session => session.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => MiningSessionStatusDbValues.FromDbValue(value));

        builder.Property(session => session.PauseReason).HasMaxLength(64);
        builder.Property(session => session.StopReason).HasMaxLength(64);

        builder.Property(session => session.StartedAt).IsRequired();
        builder.Property(session => session.LastActivityAt).IsRequired();

        builder.HasOne<UserHardware>()
            .WithMany()
            .HasForeignKey(session => session.UserHardwareId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<Pool>()
            .WithMany()
            .HasForeignKey(session => session.PoolId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(session => session.CoinId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne<MiningAlgo>()
            .WithMany()
            .HasForeignKey(session => session.MiningAlgoId)
            .OnDelete(DeleteBehavior.Restrict);

        // At most one in-flight session per device. Enforced by a partial unique index so
        // stopped sessions accumulate freely as history.
        builder.HasIndex(session => session.UserHardwareId)
            .IsUnique()
            .HasFilter("status IN ('running', 'paused')");

        builder.HasIndex(session => new { session.Status, session.LastActivityAt });
    }
}

internal sealed class MiningLogConfiguration : IEntityTypeConfiguration<MiningLog>
{
    public void Configure(EntityTypeBuilder<MiningLog> builder)
    {
        builder.HasKey(log => log.Id);

        builder.Property(log => log.EventType)
            .IsRequired()
            .HasMaxLength(64)
            .HasConversion(
                eventType => eventType.ToDbValue(),
                value => MiningEventTypeDbValues.FromDbValue(value));

        builder.Property(log => log.Reason).HasMaxLength(256);

        // Event-specific detail varies per event type, so it is stored as free-form JSON.
        builder.Property(log => log.Metadata).HasColumnType("jsonb");

        builder.Property(log => log.CreatedAt).IsRequired();

        builder.HasOne<User>()
            .WithMany()
            .HasForeignKey(log => log.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<UserHardware>()
            .WithMany()
            .HasForeignKey(log => log.UserHardwareId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<UserHardwareMiner>()
            .WithMany()
            .HasForeignKey(log => log.UserHardwareMinerId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<Pool>()
            .WithMany()
            .HasForeignKey(log => log.PoolId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(log => log.CoinId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasIndex(log => new { log.UserId, log.CreatedAt });
        builder.HasIndex(log => log.UserHardwareMinerId);
    }
}
