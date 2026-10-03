using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace TokenMiner.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class MakeHardwareIdUnique : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "ix_user_hardware_user_id_hardware_id",
                table: "user_hardware");

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_hardware_id",
                table: "user_hardware",
                column: "hardware_id",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_user_id",
                table: "user_hardware",
                column: "user_id");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "ix_user_hardware_hardware_id",
                table: "user_hardware");

            migrationBuilder.DropIndex(
                name: "ix_user_hardware_user_id",
                table: "user_hardware");

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_user_id_hardware_id",
                table: "user_hardware",
                columns: new[] { "user_id", "hardware_id" },
                unique: true);
        }
    }
}
