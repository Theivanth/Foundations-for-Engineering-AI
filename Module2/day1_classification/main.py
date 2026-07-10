from label_generator import (
    load_data,
    add_status_column,
    get_dangerous_days
)

from visualize import plot_status_distribution

from alerts import (
    generate_alerts,
    generate_report
)


def main():

    df = load_data(filename="environment.csv")
    df = add_status_column(df)

    print(df)

    danger = get_dangerous_days(df)

    print(danger)

    plot_status_distribution(df)

    generate_alerts(df)

    generate_report(df)


if __name__ == "__main__":
    main()