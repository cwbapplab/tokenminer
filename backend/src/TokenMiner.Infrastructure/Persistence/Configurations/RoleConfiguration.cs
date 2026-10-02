using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using TokenMiner.Domain.Users;

namespace TokenMiner.Infrastructure.Persistence.Configurations;

internal sealed class RoleConfiguration : IEntityTypeConfiguration<Role>
{
    public void Configure(EntityTypeBuilder<Role> builder)
    {
        builder.HasKey(role => role.Id);

        builder.Property(role => role.Name)
            .IsRequired()
            .HasMaxLength(64);

        builder.HasIndex(role => role.Name).IsUnique();

        builder.HasData(
            new { Id = RoleIds.User, Name = Role.User },
            new { Id = RoleIds.Admin, Name = Role.Admin });
    }
}
