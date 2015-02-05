/**
 * Copyright 2015 Brigham Young University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.byu.nlp.crowdsourcing.app;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.vfs2.FileSystemException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;

import edu.byu.nlp.data.docs.CountCutoffFeatureSelectorFactory;
import edu.byu.nlp.data.docs.FeatureSelectorFactories;
import edu.byu.nlp.data.docs.JSONDocumentDatasetBuilder;
import edu.byu.nlp.data.docs.TokenizerPipes;
import edu.byu.nlp.data.docs.TopNPerDocumentFeatureSelectorFactory;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.io.Files2;
import edu.byu.nlp.io.Paths;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Matrices;
import edu.byu.nlp.util.jargparser.ArgumentParser;
import edu.byu.nlp.util.jargparser.annotations.Option;

/**
 * @author plf1
 *
 * Read in crowdflower data, output a file containing fitted 
 * annotator confusion matrices (for use in future simulations)
 */
public class CrowdflowerAnnotatorParameterEstimator {
  private static Logger logger = LoggerFactory.getLogger(CrowdflowerAnnotatorParameterEstimator.class);

  @Option(help = "A json annotation stream containing annotations to be fitted.")
  private static String jsonStream = "/aml/data/plf1/cfgroups/cfgroups1000.json"; 

  @Option(help = "The file where fitted confusion matrices should be written to (in json)")
  private static String output = null; 
  
  public enum AggregationMethod {NONE, KMEANS, ACCURACY, RANDOM} 
  @Option(help = "")
  private static AggregationMethod aggregate = AggregationMethod.NONE;

  @Option(help = "The max number of buckets to aggregate annotators into.")
  private static int k = 5;

  @Option(help = "added to annotation counts before normalizing to avoid zero entries")
  private static double smooth = 0.01;

  @Option(help = "how many iterations should clustering algorithms do?")
  private static int maxIterations = 10000;
  
  
  public static void main(String[] args) throws IOException{
    // parse CLI arguments
    new ArgumentParser(CrowdflowerAnnotatorParameterEstimator.class).parseArgs(args);
    Preconditions.checkNotNull(jsonStream,"You must provide a valid --json-stream!");
    Preconditions.checkArgument(smooth>=0,"invalid smoothing value="+smooth);
    Preconditions.checkArgument(k>0,"invalid number of clusters="+k);
    
    // compile annotation stream data into per-annotator annotation matrices
    Dataset data = readData(jsonStream, new MersenneTwister(1));
    logger.info("dataset="+data);
    
    // aggregate them
    int[][][] annotations = Datasets.confusionMatrices(data); // annotations[annotator][true label][annotation] = count
    double[][][] confusions = aggregateAnnotatorsByConfusionMatrix(annotations, aggregate, k, maxIterations, smooth);

    // output to console 
    logger.info("annotators=\n"+Matrices.toString(confusions, 10, 10, 20, 3));
    for (int i=0; i<confusions.length; i++){
      logger.info("annotator #"+i+" accuracy="+accuracyOf(confusions[i]));
    }
    
    // output to file 
    if (output!=null){
      Files2.write(Matrices.toString(confusions), output);
    }
  }
  /////////////////////////////
  // END MAIN
  /////////////////////////////


  public static double[][][] aggregateAnnotatorsByConfusionMatrix(int[][][] annotations, AggregationMethod aggregate, int k, int maxIterations, double smooth) throws IOException {
    Preconditions.checkNotNull(annotations);
    Preconditions.checkArgument(annotations.length>0);
    int numAnnotators = annotations.length;
    
    logger.info("num annotators="+numAnnotators);
    logger.info("total annotations="+IntArrays.sum(annotations));

    // smoothed annotation counts
    double[][][] confusions = Matrices.fromInts(annotations);
    Matrices.addToSelf(confusions, smooth);

    // empirical confusion matrices
    Matrices.normalizeRowsToSelf(confusions);
    
    // put each annotator in a singleton cluster
    int[] clusters = IntArrays.sequence(0, confusions.length);
    
    // precompute potenially useful quantities
    double uniformClusterSize = (double)numAnnotators / k;  

    
    // transformed confusion matrices
    switch(aggregate){
    case NONE:
      break;
    case RANDOM:
      // add (approx) equal shares of each cluster to the vector, then shuffle 
      Arrays.fill(clusters, k); // this ensures leftovers are assigned to last cluster
      for (int c=0; c<k; c++){
        Arrays.fill(clusters, (int)Math.floor(c*uniformClusterSize), (int)Math.floor(c*uniformClusterSize+uniformClusterSize), c);
      }
      clusters = IntArrays.shuffled(clusters);
      break;
    case ACCURACY:
      sortByAccuracy(confusions); // re-order annotators so that more accurate ones appear first
      logger.debug("sorting annotators by accuracy");
      for (int i=0; i<confusions.length; i++){
        logger.debug("annotator #"+i+" accuracy="+accuracyOf(confusions[i]));
      }
      
      // now divide annotators into equal chunks--like accuracies will cluster together
      Arrays.fill(clusters, k); // this ensures leftovers are assigned to last cluster
      for (int c=0; c<k; c++){
        Arrays.fill(clusters, (int)Math.floor(c*uniformClusterSize), (int)Math.floor(c*uniformClusterSize+uniformClusterSize), c);
      }
      break;
    case KMEANS:
      assignKMeansClusters(confusions, clusters, k, maxIterations);
      break;
    default:
      throw new IllegalArgumentException("unknown aggregation method="+aggregate);
    }
    
    // collect clusters into sets
    Map<Integer,Set<double[][]>> clusterMap = Maps.newIdentityHashMap();
    for (int i=0; i<clusters.length; i++){
      int clusterAssignment = clusters[i];
      if (!clusterMap.containsKey(clusterAssignment)){
        clusterMap.put(clusterAssignment, Sets.<double[][]>newIdentityHashSet());
      }
      clusterMap.get(clusterAssignment).add(confusions[i]);
    }
    
    // aggregate cluster sets
    List<double[][]> clusteredConfusions = Lists.newArrayList();
    for (Set<double[][]> cluster: clusterMap.values()){
      double[][][] clusterTensor = cluster.toArray(new double[][][]{});
      double[][] averagedConfusions = Matrices.sumOverFirst(clusterTensor);
      Matrices.divideToSelf(averagedConfusions, cluster.size());
      clusteredConfusions.add(averagedConfusions);
    }
    
    // re-assign confusions
    return clusteredConfusions.toArray(new double[][][]{});
    
  }
  
  
  /**
   * This returns a set of clustered matrices. Averaging them yields the centroid of the cluster.  
   */
  private static void assignKMeansClusters(double[][][] confusions, final int[] clusterAssignments, int k, int maxIterations){
    Preconditions.checkNotNull(confusions);
    Preconditions.checkNotNull(clusterAssignments);
    Preconditions.checkArgument(confusions.length>0);
    Preconditions.checkArgument(confusions.length==clusterAssignments.length);
    int numClasses = confusions[0].length;
    
    List<ClusterableAnnotator> clusterableAnnotators= Lists.newArrayList();
    for (double[][] confusion: confusions){
      clusterableAnnotators.add(new ClusterableAnnotator(confusion));
    }
    
    KMeansPlusPlusClusterer<ClusterableAnnotator> clusterer = new KMeansPlusPlusClusterer<ClusterableAnnotator>(k, maxIterations);
    List<CentroidCluster<ClusterableAnnotator>> clusterCentroids = clusterer.cluster(clusterableAnnotators);
    
    int annotatorIndex = 0;
    for (int c=0; c<clusterCentroids.size(); c++){
      for (ClusterableAnnotator annotator: clusterCentroids.get(c).getPoints()){
        // note: we don't return the centroid point here because averaging the points in the cluster 
        // yields precisely the centroid point.
        // stick this annotator in this location in the confusions
        confusions[annotatorIndex] = Matrices.unflatten(annotator.getPoint(), numClasses, numClasses);
        // assign this position to cluster c
        clusterAssignments[annotatorIndex] = c;
        annotatorIndex++;
      }
    }
    
  }
  
  
  private static class ClusterableAnnotator implements Clusterable{
    private double[] confusionVector;
    public ClusterableAnnotator(double[][] annotatorConfusion){
      this.confusionVector=Matrices.flatten(annotatorConfusion);
    }
    @Override
    public double[] getPoint() {
      return confusionVector;
    }
  }
  
  private static void sortByAccuracy(double[][][] confusions){
    Arrays.sort(confusions, new Comparator<double[][]>() {
      @Override
      public int compare(double[][] o1, double[][] o2) {
        double acc1 = accuracyOf(o1);
        double acc2 = accuracyOf(o2);
        return Double.compare(acc2, acc1); // high-to-low
      }
    });
  }
  
  /**
   * average diagonal value
   */
  private static double accuracyOf(double[][] annotatorConfusion){
    return Matrices.trace(annotatorConfusion)/annotatorConfusion.length;
  }
  

  private static Dataset readData(String jsonStream, RandomGenerator rnd) throws FileSystemException, FileNotFoundException {
    // these parameters are not important since we will ignore the data itself and concentrate only on annotations
    // in this script
    int featureCountCutoff = -1;
    int topNFeaturesPerDocument = -1;
    Integer featureNormalizer = null;
    Function<String, String> docTransform = null;
    Function<List<String>, List<String>> tokenTransform = null;
    
    // data reader pipeline per dataset
    // build a dataset, doing all the tokenizing, stopword removal, and feature normalization
    String folder = Paths.directory(jsonStream);
    String file = Paths.baseName(jsonStream);
    Dataset data = new JSONDocumentDatasetBuilder(folder, file, 
          docTransform, TokenizerPipes.McCallumAndNigam(), tokenTransform, 
          FeatureSelectorFactories.conjoin(
              new CountCutoffFeatureSelectorFactory<String>(featureCountCutoff), 
              (topNFeaturesPerDocument<0)? null: new TopNPerDocumentFeatureSelectorFactory<String>(topNFeaturesPerDocument)),
          featureNormalizer, rnd)
          .dataset();
      
    // Postprocessing: remove all documents with duplicate sources or empty features
    data = Datasets.filteredDataset(data, Predicates.and(Datasets.filterDuplicateSources(), Datasets.filterNonEmpty()));
    
    logger.info("Number of labeled instances = " + data.getInfo().getNumDocumentsWithObservedLabels());
    logger.info("Number of unlabeled instances = " + data.getInfo().getNumDocumentsWithoutObservedLabels());
    logger.info("Number of tokens = " + data.getInfo().getNumTokens());
    logger.info("Number of features = " + data.getInfo().getNumFeatures());
    logger.info("Number of classes = " + data.getInfo().getNumClasses());
    logger.info("Average Document Size = " + (data.getInfo().getNumTokens()/data.getInfo().getNumDocuments()));

    return data;
  }
  
}
