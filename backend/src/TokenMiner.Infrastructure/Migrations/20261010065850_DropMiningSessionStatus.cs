using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace TokenMiner.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class DropMiningSessionStatus : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "ix_user_hardware_miners_status_last_activity_at",
                table: "user_hardware_miners");

            migrationBuilder.DropIndex(
                name: "ix_user_hardware_miners_user_hardware_id",
                table: "user_hardware_miners");

            migrationBuilder.DropColumn(
                name: "pause_reason",
                table: "user_hardware_miners");

            migrationBuilder.DropColumn(
                name: "paused_at",
                table: "user_hardware_miners");

            migrationBuilder.DropColumn(
                name: "status",
                table: "user_hardware_miners");

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_miners_user_hardware_id",
                table: "user_hardware_miners",
                column: "user_hardware_id",
                unique: true,
                filter: "stopped_at IS NULL");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "ix_user_hardware_miners_user_hardware_id",
                table: "user_hardware_miners");

            migrationBuilder.AddColumn<string>(
                name: "pause_reason",
                table: "user_hardware_miners",
                type: "character varying(64)",
                maxLength: 64,
                nullable: true);

            migrationBuilder.AddColumn<DateTimeOffset>(
                name: "paused_at",
                table: "user_hardware_miners",
                type: "timestamp with time zone",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "status",
                table: "user_hardware_miners",
                type: "character varying(32)",
                maxLength: 32,
                nullable: false,
                defaultValue: "");

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_miners_status_last_activity_at",
                table: "user_hardware_miners",
                columns: new[] { "status", "last_activity_at" });

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_miners_user_hardware_id",
                table: "user_hardware_miners",
                column: "user_hardware_id",
                unique: true,
                filter: "status IN ('running', 'paused')");
        }
    }
}
