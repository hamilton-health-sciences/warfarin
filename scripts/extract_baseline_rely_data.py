import pandas as pd

import psycopg2 as pg


def main(args):
    # Connect to database
    conn = pg.connect(host=args.db_ip,
                      port=args.db_port,
                      dbname=args.db_name,
                      user=args.db_username,
                      password=args.db_password)
    conn.cursor().execute("set search_path to rely;")

    # Read out baseline drug data
    drugs = pd.read_sql(
        "SELECT id AS subjid, cmbbb AS beta_blocker, cmasab AS asa, "
        "cmaceb AS ace, cmstatb AS statin, isigndt AS date "
        "FROM rely.concomitant_medications_report_9",
        conn
    ).set_index("subjid")

    drugs.to_csv(args.output_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db_ip", type=str, required=True)
    parser.add_argument("--db_port", type=str, required=True)
    parser.add_argument("--db_name", type=str, required=True)
    parser.add_argument("--db_username", type=str, required=True)
    parser.add_argument("--db_password", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    args = parser.parse_args()

    main(args)
