"""GitHub Actions CI adapter for KCS.

Integrates KCS kernel analysis into GitHub Actions workflows.
Provides automated impact analysis for pull requests.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class GitHubContext:
    """GitHub Actions context information."""

    repository: str
    ref: str
    sha: str
    event_name: str
    actor: str
    workflow: str
    job: str
    run_id: str
    run_number: str
    api_url: str = "https://api.github.com"
    token: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GitHubContext":
        """Create GitHub context from environment variables."""
        return cls(
            repository=os.getenv("GITHUB_REPOSITORY", ""),
            ref=os.getenv("GITHUB_REF", ""),
            sha=os.getenv("GITHUB_SHA", ""),
            event_name=os.getenv("GITHUB_EVENT_NAME", ""),
            actor=os.getenv("GITHUB_ACTOR", ""),
            workflow=os.getenv("GITHUB_WORKFLOW", ""),
            job=os.getenv("GITHUB_JOB", ""),
            run_id=os.getenv("GITHUB_RUN_ID", ""),
            run_number=os.getenv("GITHUB_RUN_NUMBER", ""),
            token=os.getenv("GITHUB_TOKEN"),
        )


@dataclass
class PullRequestChange:
    """Represents a change in a pull request."""

    filename: str
    status: str  # added, removed, modified
    additions: int
    deletions: int
    changes: int
    patch: Optional[str] = None


@dataclass
class ImpactAnalysisResult:
    """Result of impact analysis."""

    pr_number: int
    base_sha: str
    head_sha: str
    changed_files: list[str]
    affected_symbols: list[str]
    risk_level: str  # low, medium, high, critical
    configs_affected: list[str]
    test_recommendations: list[str]
    review_recommendations: list[str]
    citations: list[dict[str, Any]]
    summary: str


class GitHubActionsAdapter:
    """Adapter for GitHub Actions integration."""

    def __init__(
        self,
        kcs_server_url: str,
        kcs_auth_token: str,
        github_context: Optional[GitHubContext] = None,
    ):
        """Initialize the adapter.

        Args:
            kcs_server_url: URL of KCS MCP server
            kcs_auth_token: Authentication token for KCS
            github_context: GitHub context (auto-detected if None)
        """
        self.kcs_server_url = kcs_server_url.rstrip("/")
        self.kcs_auth_token = kcs_auth_token
        self.github_context = github_context or GitHubContext.from_env()

        # Setup session for KCS API calls
        self.kcs_session = requests.Session()
        self.kcs_session.headers.update(
            {
                "Authorization": f"Bearer {kcs_auth_token}",
                "Content-Type": "application/json",
            }
        )

        # Setup session for GitHub API calls
        self.github_session = requests.Session()
        if self.github_context.token:
            self.github_session.headers.update(
                {
                    "Authorization": f"token {self.github_context.token}",
                    "Accept": "application/vnd.github.v3+json",
                }
            )

    def get_pr_changes(self, pr_number: int) -> list[PullRequestChange]:
        """Get changes from a pull request.

        Args:
            pr_number: Pull request number

        Returns:
            List of changes in the PR
        """
        url = f"{self.github_context.api_url}/repos/{self.github_context.repository}/pulls/{pr_number}/files"

        try:
            response = self.github_session.get(url)
            response.raise_for_status()

            changes = []
            for file_data in response.json():
                change = PullRequestChange(
                    filename=file_data["filename"],
                    status=file_data["status"],
                    additions=file_data["additions"],
                    deletions=file_data["deletions"],
                    changes=file_data["changes"],
                    patch=file_data.get("patch"),
                )
                changes.append(change)

            return changes

        except requests.RequestException as e:
            logger.error(f"Failed to get PR changes: {e}")
            raise

    def get_pr_diff(self, pr_number: int) -> str:
        """Get full diff for a pull request.

        Args:
            pr_number: Pull request number

        Returns:
            Full diff content
        """
        url = f"{self.github_context.api_url}/repos/{self.github_context.repository}/pulls/{pr_number}"

        try:
            response = self.github_session.get(
                url, headers={"Accept": "application/vnd.github.v3.diff"}
            )
            response.raise_for_status()
            return response.text

        except requests.RequestException as e:
            logger.error(f"Failed to get PR diff: {e}")
            raise

    def call_kcs_api(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Call KCS MCP API endpoint.

        Args:
            endpoint: API endpoint (e.g., '/mcp/tools/impact_of')
            data: Request payload

        Returns:
            API response data
        """
        url = f"{self.kcs_server_url}{endpoint}"

        try:
            response = self.kcs_session.post(url, json=data)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"KCS API call failed: {e}")
            raise

    def analyze_pr_impact(self, pr_number: int) -> ImpactAnalysisResult:
        """Analyze impact of a pull request.

        Args:
            pr_number: Pull request number

        Returns:
            Impact analysis result
        """
        logger.info(f"Analyzing impact for PR #{pr_number}")

        # Get PR information
        pr_url = f"{self.github_context.api_url}/repos/{self.github_context.repository}/pulls/{pr_number}"
        pr_response = self.github_session.get(pr_url)
        pr_response.raise_for_status()
        pr_data = pr_response.json()

        base_sha = pr_data["base"]["sha"]
        head_sha = pr_data["head"]["sha"]

        # Get changes and diff
        changes = self.get_pr_changes(pr_number)
        diff_content = self.get_pr_diff(pr_number)

        # Filter for kernel source files
        kernel_files = [
            change.filename
            for change in changes
            if self._is_kernel_source_file(change.filename)
        ]

        if not kernel_files:
            return ImpactAnalysisResult(
                pr_number=pr_number,
                base_sha=base_sha,
                head_sha=head_sha,
                changed_files=[],
                affected_symbols=[],
                risk_level="low",
                configs_affected=[],
                test_recommendations=[],
                review_recommendations=["No kernel source files changed"],
                citations=[],
                summary="This PR does not modify kernel source code",
            )

        # Call KCS impact analysis
        impact_data = self.call_kcs_api(
            "/mcp/tools/impact_of",
            {"diff": diff_content, "files": kernel_files, "config": "x86_64:defconfig"},
        )

        # Determine risk level based on impact
        risk_level = self._calculate_risk_level(impact_data, changes)

        # Generate recommendations
        test_recommendations = self._generate_test_recommendations(
            impact_data, kernel_files
        )
        review_recommendations = self._generate_review_recommendations(
            impact_data, changes
        )

        # Create summary
        summary = self._create_impact_summary(impact_data, kernel_files, risk_level)

        return ImpactAnalysisResult(
            pr_number=pr_number,
            base_sha=base_sha,
            head_sha=head_sha,
            changed_files=kernel_files,
            affected_symbols=impact_data.get("modules", []),
            risk_level=risk_level,
            configs_affected=impact_data.get("configs", []),
            test_recommendations=test_recommendations,
            review_recommendations=review_recommendations,
            citations=impact_data.get("cites", []),
            summary=summary,
        )

    def post_pr_comment(self, pr_number: int, comment: str) -> bool:
        """Post a comment on a pull request.

        Args:
            pr_number: Pull request number
            comment: Comment content (Markdown supported)

        Returns:
            True if successful, False otherwise
        """
        url = f"{self.github_context.api_url}/repos/{self.github_context.repository}/issues/{pr_number}/comments"

        try:
            response = self.github_session.post(url, json={"body": comment})
            response.raise_for_status()
            logger.info(f"Posted comment on PR #{pr_number}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to post PR comment: {e}")
            return False

    def format_impact_comment(self, result: ImpactAnalysisResult) -> str:
        """Format impact analysis result as GitHub comment.

        Args:
            result: Impact analysis result

        Returns:
            Formatted Markdown comment
        """
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}

        emoji = risk_emoji.get(result.risk_level, "⚪")

        comment = f"""## {emoji} Kernel Impact Analysis

**Risk Level:** {result.risk_level.upper()}

**Summary:** {result.summary}

### Changed Files
"""

        for file in result.changed_files:
            comment += f"- `{file}`\n"

        if result.affected_symbols:
            comment += "\n### Affected Symbols\n"
            for symbol in result.affected_symbols[:10]:  # Limit to first 10
                comment += f"- `{symbol}`\n"
            if len(result.affected_symbols) > 10:
                comment += f"- ... and {len(result.affected_symbols) - 10} more\n"

        if result.configs_affected:
            comment += "\n### Affected Configurations\n"
            for config in result.configs_affected:
                comment += f"- `{config}`\n"

        if result.test_recommendations:
            comment += "\n### Test Recommendations\n"
            for rec in result.test_recommendations:
                comment += f"- {rec}\n"

        if result.review_recommendations:
            comment += "\n### Review Recommendations\n"
            for rec in result.review_recommendations:
                comment += f"- {rec}\n"

        if result.citations:
            comment += "\n### Citations\n"
            for citation in result.citations[:5]:  # Limit to first 5
                if isinstance(citation, dict) and "path" in citation:
                    comment += f"- [`{citation['path']}:{citation['start']}`]"
                    if "sha" in citation:
                        comment += f" (@{citation['sha'][:8]})"
                    comment += "\n"

        comment += "\n---\n*Analysis powered by [Kernel Context Server](https://github.com/kernel-context-server)*"

        return comment

    def run_action(self, action: str, **kwargs) -> dict[str, Any]:
        """Run a specific CI action.

        Args:
            action: Action to run ('analyze_pr', 'check_symbols', etc.)
            **kwargs: Action-specific parameters

        Returns:
            Action result
        """
        if action == "analyze_pr":
            pr_number = kwargs.get("pr_number")
            if not pr_number:
                raise ValueError("pr_number required for analyze_pr action")

            result = self.analyze_pr_impact(pr_number)

            # Post comment if requested
            if kwargs.get("post_comment", True):
                comment = self.format_impact_comment(result)
                self.post_pr_comment(pr_number, comment)

            return asdict(result)

        elif action == "check_symbols":
            symbols = kwargs.get("symbols", [])
            results = {}

            for symbol in symbols:
                try:
                    symbol_data = self.call_kcs_api(
                        "/mcp/tools/get_symbol", {"symbol": symbol}
                    )
                    results[symbol] = symbol_data
                except Exception as e:
                    results[symbol] = {"error": str(e)}

            return {"symbols": results}

        else:
            raise ValueError(f"Unknown action: {action}")

    def _is_kernel_source_file(self, filename: str) -> bool:
        """Check if file is a kernel source file."""
        kernel_extensions = {".c", ".h", ".S", ".s", ".asm"}
        kernel_paths = {
            "arch/",
            "block/",
            "crypto/",
            "drivers/",
            "fs/",
            "include/",
            "init/",
            "ipc/",
            "kernel/",
            "lib/",
            "mm/",
            "net/",
            "security/",
            "sound/",
            "tools/",
            "usr/",
            "virt/",
        }

        ext = Path(filename).suffix.lower()
        if ext not in kernel_extensions:
            return False

        # Check if in kernel path
        return any(filename.startswith(path) for path in kernel_paths)

    def _calculate_risk_level(
        self, impact_data: dict[str, Any], changes: list[PullRequestChange]
    ) -> str:
        """Calculate risk level based on impact analysis."""
        risk_factors = 0

        # Check for high-risk areas
        high_risk_paths = {"kernel/", "mm/", "arch/", "drivers/"}
        for change in changes:
            if any(change.filename.startswith(path) for path in high_risk_paths):
                risk_factors += 2

        # Check for large changes
        total_changes = sum(change.changes for change in changes)
        if total_changes > 500:
            risk_factors += 2
        elif total_changes > 100:
            risk_factors += 1

        # Check impact analysis results
        if impact_data.get("risks"):
            risk_factors += len(impact_data["risks"])

        configs_affected = len(impact_data.get("configs", []))
        if configs_affected > 5:
            risk_factors += 2
        elif configs_affected > 2:
            risk_factors += 1

        # Determine risk level
        if risk_factors >= 6:
            return "critical"
        elif risk_factors >= 4:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"

    def _generate_test_recommendations(
        self, impact_data: dict[str, Any], changed_files: list[str]
    ) -> list[str]:
        """Generate test recommendations based on impact."""
        recommendations = []

        # Check for affected modules
        modules = impact_data.get("modules", [])
        if modules:
            recommendations.append(f"Test affected modules: {', '.join(modules[:3])}")

        # Check for test files
        test_files = impact_data.get("tests", [])
        if test_files:
            recommendations.append(f"Run existing tests: {', '.join(test_files[:3])}")

        # File-specific recommendations
        for file in changed_files:
            if "mm/" in file:
                recommendations.append("Run memory management stress tests")
            elif "fs/" in file:
                recommendations.append("Run filesystem regression tests")
            elif "net/" in file:
                recommendations.append("Run network stack tests")
            elif "drivers/" in file:
                recommendations.append("Test on affected hardware configurations")

        return recommendations[:5]  # Limit recommendations

    def _generate_review_recommendations(
        self, impact_data: dict[str, Any], changes: list[PullRequestChange]
    ) -> list[str]:
        """Generate code review recommendations."""
        recommendations = []

        # Check for maintainers
        owners = impact_data.get("owners", [])
        if owners:
            recommendations.append(
                f"Request review from maintainers: {', '.join(owners[:3])}"
            )

        # Check for risks
        risks = impact_data.get("risks", [])
        if "holds_spinlock" in risks:
            recommendations.append(
                "Review for proper spinlock usage and deadlock prevention"
            )
        if "in_irq_context" in risks:
            recommendations.append(
                "Verify interrupt context safety and atomic operations"
            )

        # Large change recommendation
        total_changes = sum(change.changes for change in changes)
        if total_changes > 200:
            recommendations.append(
                "Consider breaking this large change into smaller commits"
            )

        return recommendations

    def _create_impact_summary(
        self, impact_data: dict[str, Any], changed_files: list[str], risk_level: str
    ) -> str:
        """Create a summary of the impact analysis."""
        file_count = len(changed_files)
        modules = impact_data.get("modules", [])
        configs = impact_data.get("configs", [])

        summary = f"This PR modifies {file_count} kernel source file"
        if file_count != 1:
            summary += "s"

        if modules:
            summary += f", affecting {len(modules)} module"
            if len(modules) != 1:
                summary += "s"
            summary += f" ({', '.join(modules[:3])})"

        if configs:
            summary += f" across {len(configs)} configuration"
            if len(configs) != 1:
                summary += "s"

        summary += f". Risk level assessed as {risk_level}."

        return summary


def main():
    """CLI entry point for GitHub Actions."""
    import argparse

    parser = argparse.ArgumentParser(description="KCS GitHub Actions CI Adapter")
    parser.add_argument(
        "action", choices=["analyze_pr", "check_symbols"], help="Action to perform"
    )
    parser.add_argument("--pr-number", type=int, help="Pull request number")
    parser.add_argument("--symbols", nargs="*", help="Symbols to check")
    parser.add_argument("--kcs-url", required=True, help="KCS server URL")
    parser.add_argument("--kcs-token", required=True, help="KCS auth token")
    parser.add_argument(
        "--post-comment",
        action="store_true",
        default=True,
        help="Post results as PR comment",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="json",
        help="Output format",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create adapter
    adapter = GitHubActionsAdapter(args.kcs_url, args.kcs_token)

    # Run action
    try:
        result = adapter.run_action(
            args.action,
            pr_number=args.pr_number,
            symbols=args.symbols or [],
            post_comment=args.post_comment,
        )

        if args.output_format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"Action {args.action} completed successfully")
            if "summary" in result:
                print(f"Summary: {result['summary']}")

    except Exception as e:
        logger.error(f"Action failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
