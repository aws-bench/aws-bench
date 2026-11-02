"""Account models are Pydantic and round-trip through JSON."""

from aws_bench.account_management.models import (
    OrgInfo,
    ScenarioAccount,
    TestEnvironment,
)


def _org() -> OrgInfo:
    return OrgInfo(
        org_id="o-abc",
        root_id="r-1",
        management_account_id="111111111111",
        management_account_email="mgmt@example.com",
    )


def _acct(scenario_name: str, account_tag: str, account_id: str) -> ScenarioAccount:
    return ScenarioAccount(
        account_id=account_id,
        email=f"{account_id}@example.com",
        scenario_name=scenario_name,
        account_tag=account_tag,
    )


def test_test_environment_round_trips_through_json():
    env = TestEnvironment(
        org=_org(),
        ou_id="ou-1",
        ou_name="my-env",
        accounts={"lambda-x": {"PRIMARY": _acct("lambda-x", "PRIMARY", "222222222222")}},
    )
    assert TestEnvironment.model_validate_json(env.model_dump_json()) == env


def test_to_scenario_account_mappings_shape():
    env = TestEnvironment(
        org=_org(),
        ou_id="ou-1",
        ou_name="my-env",
        accounts={
            "lambda-x": {"PRIMARY": _acct("lambda-x", "PRIMARY", "222222222222")},
            "lambda-y": {"PRIMARY": _acct("lambda-y", "PRIMARY", "333333333333")},
        },
    )
    assert env.to_scenario_account_mappings() == {
        "lambda-x": {"PRIMARY": "222222222222"},
        "lambda-y": {"PRIMARY": "333333333333"},
    }


def test_mapping_for_returns_tag_to_account_id():
    env = TestEnvironment(
        org=_org(),
        ou_id="ou-1",
        ou_name="my-env",
        accounts={
            "lambda-x": {
                "PRIMARY": _acct("lambda-x", "PRIMARY", "222222222222"),
                "SECONDARY": _acct("lambda-x", "SECONDARY", "444444444444"),
            }
        },
    )
    assert env.mapping_for("lambda-x") == {"PRIMARY": "222222222222", "SECONDARY": "444444444444"}
    assert env.mapping_for("nope") == {}
