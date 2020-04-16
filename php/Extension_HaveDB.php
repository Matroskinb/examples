<?php


namespace led\Module\Model\Extension;

use led\Log\Log;
use led\DataProviders\DB;
use led\Helpers\Arrays as Arr;

/**
 * Trait HaveDB used to add possibilities
 * for using database through DB data provider
 * in led framework.
 *
 * used with haveState module
 *
 * @package led\Module\Model\Extension
 */
trait HaveDB
{

    /**
     * declare model as db table for use update, insert, delete queries
     * @var string table related to model
     */
    protected static $table;

    /**
     * primary key in table, value of this attribute must be declared as protected property
     * @var string
     */
    protected static $primary = 'id';

    /**
     * default primary attribute, used as example
     * @var bool
     */
    public $id = false;

    /**
     * is model inserted in database
     * @var bool
     */
    protected $__isInserted = false;

    /**
     * is model updated in database
     * @var bool
     */
    protected $__isUpdated = false;

    /**
     * detects if model deleted from database
     * @var bool
     */
    protected $__isDeleted = false;

    /**
     * fetch row from database
     * @param $query
     * @param $args
     * @return mixed
     */
    protected static function fetch($query, $args = [])
    {
        $query = DB::query($query, $args);
        $query->setFetchMode(\PDO::FETCH_CLASS, static::class);
        $query->execute();
        $res = $query->fetch();

        if (is_bool($res) && !$res) {
            return null;
        }

        return $res;
    }

    /**
     * fetch rows from database
     * @param $query
     * @param $args
     * @return array
     */
    protected static function fetchAll($query, $args = []): array
    {
        return DB::query($query, $args)
            ->fetchAll(\PDO::FETCH_CLASS, static::class);
    }

    /**
     * save instance in database
     * @param array $fields
     * @return HaveDB
     * @throws \Exception
     */
    public function save(array $fields = []): self
    {
        if ($this->isInserted()) {
            return $this->update($fields);
        }

        return $this->insert();
    }

    /**
     * is model inserted in database
     * @return bool
     */
    public function isInserted(): bool
    {
        return $this->__isInserted;
    }

    /**
     * update instance database
     * @param array $fields - if strict update over fields needed
     * @return HaveDB
     * @throws \Exception
     */
    public function update(array $fields = []): self
    {
        $this->__isUpdated = false;
        $values = $this->getDBArray($fields);

        if (!$this->getPrimaryValue()) {
            $this->log('error', 'Model have no primary key set');
            return $this;
        }

        if ($this->isDeleted()) {
            $this->log('error', 'trying to update deleted model');
            return $this;
        }

        $this->log('info', 'updating', ['primary' => $this->getPrimaryValue()]);

        $result = DB::update($this::$table, $values, [$this::$primary => $this->getPrimaryValue()]);

        if (is_int($result)) {
            $this->log('info', 'updated model', ['primary' => $this->getPrimaryValue()]);
            $this->setUpdated();
        } else if(is_bool($result) && !$result) {
            $this->log('error', 'model not updated');
        } else {
            $this->log('error', 'some scared happens');
        }


        return $this;
    }

    /**
     * get array values specially prepared for database table
     * some attributes mapping for mysql database
     * @param array $fields
     * @return array
     */
    public function getDBArray(array $fields = []): array
    {
        $values = $this->getState();

        if (count($fields) > 0) {
            $values = array_intersect_key(array_flip($fields), $values);
        }
        // prepare values for storing to database
        foreach ($values as &$val) {
            if (is_bool($val)) {
                $val = (int)$val;
            }
        }

        return $values;
    }

    /**
     * returns primary attribute of model
     * @return mixed
     */
    public function getPrimaryValue()
    {
        return $this->{$this::$primary};
    }

    protected function log(string $level, string $message, array $extra = [], bool $logAttributes = true): self
    {
        $extra = $logAttributes ? array_merge($this->getDBArray(), $extra) : $extra;
        Log::$level("$message in [{$this::$table}]", $extra);
        return $this;
    }

    /**
     * is model deleted from database
     * @return bool
     */
    public function isDeleted(): bool
    {
        return $this->__isDeleted;
    }

    /**
     * set updated flag if model updated in DB
     * @return $this
     */
    protected function setUpdated(): self
    {
        $this->__isUpdated = true;

        return $this;
    }

    /**
     * insert instance in database
     */

    public function insert(): self
    {
        $fields = $this->getDBArray();
        $this->log('info', 'inserting model');

        $insertedId = DB::insert($this::$table, $fields);

        if (!$insertedId) {
            $this->log('error', 'cant insert model');
            $this->id = false;
            return $this;
        }

        $this->{$this::$primary} = $insertedId;

        $this->setInserted();
        $this->log('info', 'inserted model', ['primary' => $insertedId]);

        return $this;
    }

    /**
     * set inserted flag if model already exists in DB
     * @return $this
     */
    protected function setInserted(): self
    {
        $this->__isInserted = true;

        return $this;
    }

    /**
     * get array of values setted in fillable attribute by default returns all properties
     * @return array
     */
    public function toArray(): array
    {
        $values = $this->getState();

        if ($this->isInserted()) {
            $values[$this::$primary] = $this->getPrimaryValue();
        }

        return $values;
    }

    /**
     * remove instance from database
     */
    public function delete(): self
    {
        if (empty($this->getPrimaryValue()) && $this->getPrimaryValue() === null) {
            $this->log('error', 'deleting without primary key');
            return $this;
        }

        $this->log('info', 'deleting', ['primary' => $this->getPrimaryValue()]);

        $result = DB::query("delete from {$this::$table} where {$this->getPrimaryKey()} = ?",
            [$this->getPrimaryValue()])->rowCount();
        $this->log('info', 'deleted model', ['primary' => $this->getPrimaryValue(), 'is_deleted' => $result]);

        if ($result) {
            $this->setDeleted();
        }

        return $this;
    }

    /**
     * returns primary key of this model from table
     * @return string
     */
    public function getPrimaryKey(): string
    {
        return $this::$primary;
    }

    /**
     * set if model deleted from database
     * @return $this
     */
    protected function setDeleted(): self
    {
        $this->setPrimaryValue(false);
        $this->__isDeleted = true;

        return $this;
    }

    /**
     * set primary key value
     * @param $value
     * @return $this
     */
    protected function setPrimaryValue($value): self
    {
        $this->{$this::$primary} = $value;

        return $this;
    }

    /**
     * is model updated in database after last update
     * @return bool
     */
    public function isUpdated(): bool
    {
        return $this->__isUpdated;
    }

    /**
     * initialisation of trait
     * @param $attributes
     * @return bool
     * @throws \Exception
     */
    protected function initializeHaveDB($attributes): bool
    {
        if (!is_array($attributes)) {
            return false;
        }

        $primary = Arr::get($attributes, $this::$primary, false);

        if (!$primary && !$this->getPrimaryValue()) {
            return false;
        }

        if ($primary) {
            $this->setPrimaryValue($primary);
        }

        $this->setInserted()->setPrimaryFillable();

        return true;
    }

    /**
     * add primary key to fillable
     * @return $this
     */
    protected function setPrimaryFillable(): self
    {
        if (!$this->isFillableLocked() && !in_array($this::$primary, $this::$__fillable, true)) {
            $this::$__fillable[] = $this::$primary;
        }

        return $this;
    }
}