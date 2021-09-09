import us.codecraft.webmagic.Page;
import us.codecraft.webmagic.Site;
import us.codecraft.webmagic.Spider;
import us.codecraft.webmagic.pipeline.JsonFilePipeline;
import us.codecraft.webmagic.processor.PageProcessor;


public class GithubRepoPageProcessor implements PageProcessor {

    private Site site = Site.me().setRetryTimes(3).setSleepTime(100);

    @Override
    public void process(Page page) {
        page.addTargetRequests(page.getHtml().links().regex("(https://github\\.com/\\w+/\\w+)").all());
        String author = page.getUrl().regex("https://github\\.com/(\\w+)/.*").toString();
        page.putField("author", author);
        String pageName = page.getHtml().xpath("//h1/strong/a/text()").toString();
        page.putField("name", pageName);
        if (page.getResultItems().get("name")==null){
            //skip this page
            page.setSkip(true);
        }
        // prevent getting banned by github.com...
        try {
            Thread.sleep(4000);
        } catch (Exception ignored) {}
        page.putField("readme", page.getHtml().xpath("//div[@id='readme']/tidyText()"));
        //System.out.println(author);
    }

    @Override
    public Site getSite() {
        return site;
    }

    public static void main(String[] args) {
        Spider
            .create(new GithubRepoPageProcessor())
           // .addUrl("https://github.com/code4craft")
            //.addUrl("https://github.com/dask")
            .addUrl("https://github.com/sindresorhus/awesome")
            .addPipeline((new JsonFilePipeline()))
            .thread(5)
            .run();
    }
}