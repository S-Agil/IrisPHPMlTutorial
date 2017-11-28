<?php
include '../vendor/autoload.php';
use Phpml\Dataset\Demo\IrisDataset;
use Phpml\CrossValidation\StratifiedRandomSplit;
use Phpml\Classification\SVC;
use Phpml\Metric\Accuracy;
use Phpml\SupportVectorMachine\Kernel;

$dataset = new IrisDataset();
$split = new StratifiedRandomSplit($dataset, 0.3, 123);

$classifier = new SVC(Kernel::RBF);
$classifier->train($split->getTrainSamples(), $split->getTrainLabels());

$predicted = $classifier->predict($split->getTestSamples());

var_dump(Accuracy::score($split->getTestLabels(), $predicted));
