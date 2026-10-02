using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Domain.Users;

namespace TokenMiner.Infrastructure.Persistence.Configurations;

internal sealed class PoolConfiguration : IEntityTypeConfiguration<Pool>
{
    public void Configure(EntityTypeBuilder<Pool> builder)
    {
        builder.HasKey(pool => pool.Id);

        builder.Property(pool => pool.SystemPoolId).IsRequired().HasMaxLength(128);
        builder.HasIndex(pool => pool.SystemPoolId).IsUnique();

        builder.Property(pool => pool.Name).IsRequired().HasMaxLength(128);

        builder.Property(pool => pool.Provider).IsRequired().HasMaxLength(64).HasDefaultValue("kryptex");

        builder.Property(pool => pool.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => MiningStatusDbValues.FromDbValue(value));

        builder.Property(pool => pool.CreatedAt).IsRequired();
        builder.Property(pool => pool.UpdatedAt).IsRequired();
    }
}

internal sealed class PoolDetailsConfiguration : IEntityTypeConfiguration<PoolDetails>
{
    public void Configure(EntityTypeBuilder<PoolDetails> builder)
    {
        builder.HasKey(details => details.Id);

        builder.Property(details => details.BaseUrl).IsRequired().HasMaxLength(512);
        builder.Property(details => details.StatusEndpoint).HasMaxLength(512);
        builder.Property(details => details.StratumEndpoint).HasMaxLength(256);

        builder.Property(details => details.PayoutMode)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                mode => mode.ToDbValue(),
                value => PayoutModeDbValues.FromDbValue(value));

        builder.Property(details => details.PayoutAddress).HasMaxLength(256);
        builder.Property(details => details.PayoutNetwork).HasMaxLength(64);

        builder.Property(details => details.CreatedAt).IsRequired();
        builder.Property(details => details.UpdatedAt).IsRequired();

        builder.HasOne<Pool>()
            .WithMany()
            .HasForeignKey(details => details.PoolId)
            .OnDelete(DeleteBehavior.Cascade);

        // One configuration row per pool.
        builder.HasIndex(details => details.PoolId).IsUnique();

        // A coin that is still referenced by a pool must not be removed silently.
        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(details => details.CoinId)
            .OnDelete(DeleteBehavior.Restrict);
    }
}

internal sealed class PoolCoinConfiguration : IEntityTypeConfiguration<PoolCoin>
{
    public void Configure(EntityTypeBuilder<PoolCoin> builder)
    {
        builder.HasKey(poolCoin => poolCoin.Id);

        builder.HasOne<Pool>()
            .WithMany()
            .HasForeignKey(poolCoin => poolCoin.PoolId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne<Coin>()
            .WithMany()
            .HasForeignKey(poolCoin => poolCoin.CoinId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasIndex(poolCoin => new { poolCoin.PoolId, poolCoin.CoinId }).IsUnique();
    }
}

internal sealed class UserHardwareConfiguration : IEntityTypeConfiguration<UserHardware>
{
    public void Configure(EntityTypeBuilder<UserHardware> builder)
    {
        builder.HasKey(hardware => hardware.Id);

        builder.Property(hardware => hardware.Name).HasMaxLength(128);

        builder.Property(hardware => hardware.Status)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                status => status.ToDbValue(),
                value => MiningStatusDbValues.FromDbValue(value));

        builder.Property(hardware => hardware.CreatedAt).IsRequired();

        builder.HasOne<User>()
            .WithMany()
            .HasForeignKey(hardware => hardware.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        // A device is identified per user by the GUID the client generates.
        builder.HasIndex(hardware => new { hardware.UserId, hardware.HardwareId }).IsUnique();
    }
}
