import os
import snscrape.modules.twitter as sntwitter
import csv
from datetime import datetime, timedelta

class Tweety():

    def get_tweets(self, keywords, tweets_per_week, weeks, lang='pl'):
        """
        :param tweets_per_week:
        :param weeks:
        :param lang:
        :param keywords: provide keywords separated by a +, e. g. "korona+szczepienie"
        :return: list Tweet objects
        """

        now = datetime.now()
        now_str = now.strftime('%Y-%m-%d')
        until = now
        since = now

        # Open/create a file to append data to
        csvFile = open(os.path.join('en', (keywords + '-sentiment-' + now_str + '.csv')), 'a', newline='',
                       encoding='utf8')

        # Use csv writer
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['id', 'date', 'tweet', 'retweet_count', 'like_count'])
        for _ in range(weeks):
            until = since
            until_str = until.strftime('%Y-%m-%d')
            since = until - timedelta(days=7)
            since_str = since.strftime('%Y-%m-%d')
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(
                    keywords + ' lang:' + lang + ' since:' + since_str + ' until:' + until_str + ' -filter:links -filter:replies').get_items()):
                if i > tweets_per_week:
                    break

                csvWriter.writerow([tweet.id, tweet.date, tweet.content, tweet.retweetCount, tweet.likeCount])
        csvFile.close()


def download_tweets_pl():
    tweety = Tweety()
    sw = ["#koronawirus+#szczepimysie", "#coronavirus+#szczepimysie", "#covid+#szczepimysie", "#covid19+#szczepimysie",
          "#covid-19+#szczepimysie", "#covid+#szczepienie",
          "#covid19+#szczepienie", "#covid-19+#szczepienie", "#koronawirus+#szczepionka", "#coronavirus+#szczepionka",
          "#covid+#szczepionka", "#covid19+#szczepionka", "#covid-19+#szczepionka"]

    tweety.get_tweets('covid+szczepimysie', 3000, 55)


def download_tweets_en():
    tweety = Tweety()  # "#coronavirus+#vaccine", "#coronavirus+#vaccination",
    sw = ["#covid19+#vaccine", "#covid19+#vaccination"]
    # "#covid-19+#vaccine", "#covid-19+#vaccination"]

    tweety.get_tweets('covid+vaccine', 500, 55, 'en')

