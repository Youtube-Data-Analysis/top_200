import requests, sys, time, os, argparse
from apscheduler.schedulers.background import BackgroundScheduler

def grab_vidz():
    # List of simple to collect features
    snippet_features = ["title",
                        "publishedAt",
                        "country"]

    # Any characters to exclude, generally these are things that become problematic in CSV files
    unsafe_characters = ['\n', '"']

    # Used to identify columns, currently hardcoded order
    header = ["channel_id"] + snippet_features + ["view_count", "sub_count", "video_Count"]


    def setup(api_path, code_path):
        with open(api_path, 'r') as file:
            api_key = file.readline()

        with open(code_path) as file:
            channelId = [x.rstrip() for x in file]

        return api_key, channelId

    api_key, channelId = setup(api_path='api_key.txt',code_path='channelId.txt')


    def prepare_feature(feature):
        # Removes any character from the unsafe characters list and surrounds the whole item in quotes
        for ch in unsafe_characters:
            feature = str(feature).replace(ch, "")
        return f'"{feature}"'


    def api_request(page_token, stuff):
        # Builds the URL and requests the JSON from it
        request_url = f"https://youtube.googleapis.com/youtube/v3/channels?part=snippet%2CcontentDetails%2Cstatistics&id={stuff}&key={api_key}"

        request = requests.get(request_url)
        if request.status_code == 429:
            print("Temp-Banned due to excess requests, please wait and continue later")
            sys.exit()
        return request.json()


    def get_tags(tags_list):
        # Takes a list of tags, prepares each tag and joins them into a string by the pipe character
        return prepare_feature("|".join(tags_list))


    def get_videos(items):
        lines = []
        for channel in items:
            comments_disabled = False
            ratings_disabled = False

            # We can assume something is wrong with the channel if it has no statistics, often this means it has been deleted
            # so we can just skip it
            if "statistics" not in channel:
                continue

            # A full explanation of all of these features can be found on the GitHub page for this project
            channel_id = prepare_feature(channel['id'])

            # Snippet and statistics are sub-dicts of channel, containing the most useful info
            snippet = channel['snippet']
            statistics = channel['statistics']
            cd = channel['contentDetails']

            # This list contains all of the features in snippet that are 1 deep and require no special processing
            features = [prepare_feature(snippet.get(feature, "")) for feature in snippet_features]

            # The following are special case features which require unique processing, or are not within the snippet dict
            view_count = statistics.get("viewCount", 0)
            comment_count = statistics.get("commentCount", 0)
            sub_count = statistics.get("subscriberCount", 0)
            video_Count = statistics.get("videoCount", 0)

            # Compiles all of the various bits of info into one consistently formatted line
            line = [channel_id] + features + [prepare_feature(x) for x in [view_count, sub_count, video_Count]]
            lines.append(",".join(line))
        return lines


    def get_pages(stuff, next_page_token="&"):
        country_data = []

        # Because the API uses page tokens (which are literally just the same function of numbers everywhere) it is much
        # more inconvenient to iterate over pages, but that is what is done here.
        while next_page_token is not None:
            # A page of data i.e. a list of videos and all needed data
            channel_data_page = api_request(next_page_token, stuff)

            # Get the next page token and build a string which can be injected into the request with it, unless it's None,
            # then let the whole thing be None so that the loop ends after this cycle
            next_page_token = channel_data_page.get("nextPageToken", None)
            next_page_token = f"&pageToken={next_page_token}&" if next_page_token is not None else next_page_token

            # Get all of the items as a list and let get_videos return the needed features
            items = channel_data_page.get('items', [])
            country_data += get_videos(items)

        return country_data


    def write_to_file(stuff, country_data):

        print(f"Writing {stuff} data to file...")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(f"{output_dir}/{time.strftime('%c')}_{stuff}_videos.csv", "w+", encoding='utf-8') as file:
            for row in country_data:
                file.write(f"{row}\n")


    def get_data():
        for stuff in channelId:
            country_data = [",".join(header)] + get_pages(stuff)
            write_to_file(stuff, country_data)


    if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument('--key_path', help='Path to the file containing the api key, by default will use api_key.txt in the same directory', default='api_key.txt')
        parser.add_argument('--country_code_path', help='Path to the file containing the list of country codes to scrape, by default will use channelId.txt in the same directory', default='channelId.txt')
        parser.add_argument('--output_dir', help='Path to save the outputted files in', default='channeloutput/')

        args = parser.parse_args()

        output_dir = args.output_dir
        api_key, stuff = setup(args.key_path, args.country_code_path)

        get_data()
    return

grab_vidz()