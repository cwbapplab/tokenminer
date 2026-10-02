using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace TokenMiner.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddStratumRouting : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "stratum_endpoint",
                table: "pool_details",
                type: "character varying(256)",
                maxLength: 256,
                nullable: true);

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_miners_worker_identifier",
                table: "user_hardware_miners",
                column: "worker_identifier");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "ix_user_hardware_miners_worker_identifier",
                table: "user_hardware_miners");

            migrationBuilder.DropColumn(
                name: "stratum_endpoint",
                table: "pool_details");
        }
    }
}
