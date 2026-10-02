using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using TokenMiner.Domain.Configuration;

namespace TokenMiner.Infrastructure.Persistence.Configurations;

internal sealed class SystemConfigurationConfiguration : IEntityTypeConfiguration<SystemConfiguration>
{
    public void Configure(EntityTypeBuilder<SystemConfiguration> builder)
    {
        builder.HasKey(setting => setting.Key);

        builder.Property(setting => setting.Key).HasMaxLength(128);

        // Values are JSON so a setting can be a scalar or a structured value.
        builder.Property(setting => setting.Value).IsRequired().HasColumnType("jsonb");

        builder.Property(setting => setting.UpdatedAt).IsRequired();
    }
}
