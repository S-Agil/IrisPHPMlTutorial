<?php

declare(strict_types=1);

namespace tests\Phpml\Metric;

use Phpml\Math\Distance\Euclidean;
use PHPUnit\Framework\TestCase;

class EuclideanTest extends TestCase
{
    /**
     * @var Euclidean
     */
    private $distanceMetric;

    public function setUp(): void
    {
        $this->distanceMetric = new Euclidean();
    }

    /**
     * @expectedException \Phpml\Exception\InvalidArgumentException
     */
    public function testThrowExceptionOnInvalidArguments(): void
    {
        $a = [0, 1, 2];
        $b = [0, 2];

        $this->distanceMetric->distance($a, $b);
    }

    public function testCalculateDistanceForOneDimension(): void
    {
        $a = [4];
        $b = [2];

        $expectedDistance = 2;
        $actualDistance = $this->distanceMetric->distance($a, $b);

        $this->assertEquals($expectedDistance, $actualDistance);
    }

    public function testCalculateDistanceForTwoDimensions(): void
    {
        $a = [4, 6];
        $b = [2, 5];

        $expectedDistance = 2.2360679774998;
        $actualDistance = $this->distanceMetric->distance($a, $b);

        $this->assertEquals($expectedDistance, $actualDistance);
    }

    public function testCalculateDistanceForThreeDimensions(): void
    {
        $a = [6, 10, 3];
        $b = [2, 5, 5];

        $expectedDistance = 6.7082039324993694;
        $actualDistance = $this->distanceMetric->distance($a, $b);

        $this->assertEquals($expectedDistance, $actualDistance);
    }
}
