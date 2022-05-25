package da.unima.ki.anyagg;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import de.unima.ki.anyburl.data.Triple;
import de.unima.ki.anyburl.Settings;
import de.unima.ki.anyburl.data.TripleSet;
import de.unima.ki.anyburl.eval.CompletionResult;
import de.unima.ki.anyburl.eval.HitsAtK;
import de.unima.ki.anyburl.eval.ResultSet;

public class AnyEgg {

	private static ArrayList<String> fileList;
	
	public static void main(String[] args2) throws IOException {
		Settings.REWRITE_REFLEXIV = false;
		
		// only files that contain this token are considered to be rankings
		String identifier = "ranking";
		
		// look at all files in these directories
		String directoryList[] = new String[]{"/path/to/folder/containing rankings with ranking_valid and rankings with ranking_test in name"};
		
		// write the resulting ranking to this destination
		String outputPath = "/path/to/final-ranking-name";
		
		// location of the train.txt, test.txt and valid.txt files of the dataset
		String dataset =  "data/codex-m";
		
		System.out.println("*** using the string \"" + identifier + "\" to identify files as ranking files");
		
		System.out.println("*** using the the following files as input to the aggregation");
		fileList = new ArrayList<String>();
		
		for (int i = 0; i < directoryList.length; i++) {
			
			File f = new File(directoryList[i]);
			// System.out.println(f);
			String[] content = f.list();
			for (int j = 0; j < content.length; j++) {
				if (content[j].contains(identifier)) {
					System.out.println("=> " + directoryList[i] + "/" + content[j]);
					fileList.add(directoryList[i] + "/" + content[j]);
				}
			}
		}
		
		
		System.out.println("*** reading evaluation datasets");
		
		TripleSet train = new TripleSet(dataset + "/train.txt");
		TripleSet test = new TripleSet(dataset + "/test.txt");
		TripleSet valid = new TripleSet(dataset + "/valid.txt");
		
		
		
		
		HashMap<String, Double> bestHeadMRR = new HashMap<String, Double>();
		HashMap<String, Double> bestTailMRR = new HashMap<String, Double>();
		HashMap<String, ResultSet> bestHeadRS = new HashMap<String, ResultSet>();
		HashMap<String, ResultSet> bestTailRS = new HashMap<String, ResultSet>();
		
		for (String relation : train.getRelations()) {
			bestHeadMRR.put(relation, -1.0);
			bestTailMRR.put(relation, -1.0);
			bestHeadRS.put(relation, null);
			bestTailRS.put(relation, null);
		}
		
		
		
		
		System.out.println("*** search for the best configuration");
		// find out which epoch is best for which relation and direction
		for (String f  : fileList) {
			if (!(f.contains("valid"))) continue;
				
			ResultSet rs = new ResultSet(f, f, true, 100);
			ResultSet rsTest = new ResultSet(f, f.replace("valid", "test"), true, 100);
			// System.out.println("read in " + rs.name);
			
			HitsAtK hitsAll = new HitsAtK();
			for (String r : valid.getRelations()) {
				ArrayList<Triple> triples = valid.getTriplesByRelation(r);
				// if (triples.size() < 100) continue;
				// if (!(r.equals("P27"))) continue;
				HitsAtK hits = new HitsAtK();
				
				
				double confAvgHeads = 0.0;
				double confAvgTails = 0.0;
				for (Triple triple : triples) {
					hits.evaluateHead(rs.getHeadCandidates(triple.toString()), triple);
					hits.evaluateTail(rs.getTailCandidates(triple.toString()), triple);
					hitsAll.evaluateHead(rs.getHeadCandidates(triple.toString()), triple);
					hitsAll.evaluateTail(rs.getTailCandidates(triple.toString()), triple);
					confAvgHeads += getAverage(rs.getHeadConfidences(triple.toString()), 10);
					confAvgTails += getAverage(rs.getTailConfidences(triple.toString()), 10);
				}
				confAvgHeads = confAvgHeads / (double)triples.size();
				confAvgTails = confAvgTails / (double)triples.size();
				// System.out.println(confAvgHeads + "\t" + confAvgTails);
				
				double mrrHead = hits.getMRRHeads();
				double mrrTail = hits.getMRRTails();
				
				// System.out.println(mrrHead + "\t" + mrrTail);
				if (bestHeadMRR.get(r) < mrrHead) {
					bestHeadMRR.put(r, mrrHead);
					bestHeadRS.put(r, rsTest);
				}
				if (bestTailMRR.get(r) < mrrTail) {
					bestTailMRR.put(r, mrrTail);
					bestTailRS.put(r, rsTest);
				}
			}
			// System.out.println();
            // System.out.println(hitsAll.getMRR() + " in " + f);
            System.out.println(hitsAll.getMRR());
		}
		
		
		System.out.println("*** apply the best configuration and create a new ranking at " + outputPath);
		PrintWriter pw = new PrintWriter(outputPath);
for (String r : test.getRelations()) {
			String r_replaced = null;
			if (!(valid.getRelations().contains(r))) {
				int max = 0;
				for (String rv : valid.getRelations()) {
					if (valid.getTriplesByRelation(rv).size() > max) {
						max = valid.getTriplesByRelation(rv).size();
						r_replaced = rv;
					}
				}
			}
			ArrayList<Triple> triples = test.getTriplesByRelation(r);
			ResultSet rsBestHead = bestHeadRS.get(r_replaced == null ? r : r_replaced);
			ResultSet rsBestTail = bestTailRS.get(r_replaced == null ? r : r_replaced);
			
			for (Triple t : triples) {
				CompletionResult crHead = rsBestHead.getCompletionResult(t.toString());
				CompletionResult crTail = rsBestTail.getCompletionResult(t.toString());
				String prediction = toJoinedCRString(t, crHead, crTail);
				pw.print(prediction);
				pw.flush();
				
			}
			
		}
		
		pw.close();
		
		
		
	}
	
	private static double getAverage(ArrayList<Double> confidences, int num) {
		double total = 0.0;
		int counter = 0;
		for (Double conf : confidences) {
			counter++;
			total += conf;
			if (counter >= num) break;
		}
		total = total / (double)num;
		// System.out.println(total);
		return total;
	}

	public static String toJoinedCRString(Triple t, CompletionResult crHead, CompletionResult crTail) {
		StringBuilder sb = new StringBuilder(t + "\n");
		sb.append("Heads: ");
		for (int i = 0; i < crHead.getHeads().size(); i++) {
			sb.append(crHead.getHeads().get(i) + "\t" + crHead.getHeadConfidences().get(i) + "\t");
		}
		sb.append("\n");
		sb.append("Tails: ");
		for (int i = 0; i < crTail.getTails().size(); i++) {
			sb.append(crTail.getTails().get(i) + "\t" + crTail.getTailConfidences().get(i) + "\t");
		}
		sb.append("\n");
		return sb.toString();
	}

}
