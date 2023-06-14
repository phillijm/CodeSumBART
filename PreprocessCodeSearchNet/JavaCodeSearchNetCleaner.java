/**
 * JavaCodeSearchNetCleaner - Formats CodeSearchNet data for CodeSumBART.
 * **This uses JavaDatasetCleaner's JavaDatasetPreprocessor:
 *          https://github.com/phillijm/JavaDatasetCleaner
 * We modified it to make "methods" and "summaries" public, not private.
 *
 * @author Jesse Phillips <j.m.phillips@lancaster.ac.uk>
 * @version 0.0.1
 **/
import uk.ac.lancs.scc.phd.jesse.JavaDatasetPreprocessor;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import org.json.*;

public class JavaCodeSearchNetCleaner
{
  /**
   * Converts data saved in CodeSearchNet format to ArrayLists.
   *
   * @param path The path to the directory where the data is.
   * @param file The filename of the file we're converting.
   * @param jDP The JavaDatasetPreprocessor we're using to process the data.
   * @return false if the data doesn't exist or can't be created, else true.
   **/
  public static boolean formatCodeSearchNetData(String path,
                                                String file,
                                                JavaDatasetPreprocessor jDP)
  {
    FileSystem fS = FileSystems.getDefault();
    try
    {
      JSONArray json;
      String data = Files.readString(fS.getPath(path + file));
      json = new JSONArray(data);
      for (Object obj: json)
      {
        JSONObject o = (JSONObject) obj;
        jDP.methods.add(o.getString("source"));
        jDP.summaries.add(o.getString("target"));
      }
    } catch (Exception e)
    {
      return false;
    }
    return true;
  }

  /**
   * Saves data in a Pseudo-JSON Format 
   * (It's not quite right - strings aren't escaped, but we'll be fixing that
   * in Python anyway).
   *
   * @param path The path to the directory where data will be stored.
   * @param jDP The JavaDatasetPreprocessor containing the data to store.
   * @return false if the files can't be created, else true.
   **/
  public static boolean saveData(String path, JavaDatasetPreprocessor jDP)
  {
    ArrayList<String> tokenisedMethods = jDP.tokeniseMethods();
    tokenisedMethods = jDP.removeRepeatData(tokenisedMethods);
    try
    {
      FileWriter fp = new FileWriter(path + "dataset.json");
      fp.write("[\n");
      for(int cnt = 0; cnt < tokenisedMethods.size() - 2; cnt++)
      {
        fp.write("  {\n");
        fp.write("    \"text\": \"" + tokenisedMethods.get(cnt) + "\",\n");
        fp.write("    \"summary\": \"" + jDP.summaries.get(cnt) + "\"\n");
        fp.write("  },\n");
      }
      int max = tokenisedMethods.size() - 1;
      fp.write("  {\n");
      fp.write("    \"text\": \"" + tokenisedMethods.get(max) + "\",\n");
      fp.write("    \"summary\": \"" + jDP.summaries.get(max) + "\"\n");
      fp.write("  }\n");
      fp.write("]\n");
      fp.close();
    } catch (IOException e)
    {
      return false;
    }
    return true;
  }

  /**
   * Runs the dataset cleaning steps on each dataset file.
   * @param path the path to the dataset file to clean.
   * @param file the name of the dataset file to clean.
   **/
  public static void runner(String path, String file)
  {
    JavaDatasetPreprocessor jDP = new JavaDatasetPreprocessor();

    jDP.setDataLocation(path);
    if (!formatCodeSearchNetData(path, file, jDP))
      return;

    jDP.trimToValidData();
    jDP.removeRepeatEntries();
    jDP.stripHTMLFromSummaries();
    jDP.extractAssumedSummaryFromJdoc();
    jDP.lowercaseSummaries();
    jDP.stripSpecialCharsFromSummaries();
    jDP.stripNewlines();

    if(!saveData(path, jDP))
      return;

    System.out.println("Good Methods: " + jDP.getNumberOfGoodMethods());
    System.out.println("Bad Methods: " + jDP.getNumberOfBadMethods());
  }

  public static void main(String[] args)
  {
    String path = "C:\\<YOUR_CODESEARCHNET_FILEPATH_GOES_HERE>\\java\\";
    String datasetFile = "pl_dataset.json";
    runner(path + "test\\", datasetFile);
    runner(path + "train\\", datasetFile);
    runner(path + "valid\\", datasetFile);
  }
}
